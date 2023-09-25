# encoding: utf-8
# @Author: Oleg Egorov, Enrico Congiu, Hector Ibarra
# @Date: Oct 28, 2022
# @Filename: projection2d.py
# @License: BSD 3-Clause
# @Copyright: Oleg Egorov, Enrico Congiu, Hector Ibarra

import os.path
import numpy as np

import lvmdatasimulator
from lvmdatasimulator import COMMON_SETUP_2D as config_2d
from lvmdatasimulator import DATA_DIR, n_process
from astropy.io import fits, ascii
from lvmdatasimulator import log
from scipy.interpolate import interp1d
from astropy.convolution import convolve
from astropy.convolution.kernels import Gaussian2DKernel
from multiprocessing import Pool
from tqdm import tqdm
from lvmdatasimulator.utils import tqdm_joblib
from joblib import Parallel, delayed, parallel_backend
from astropy import units as u
from astropy.time import Time

def spec_fragment_convolve_psf(spec_oversampled_cut=None, xpos_oversampled_cut=None, xpos_ccd=None, ypos_ccd=None,
                               focus=None, convolve_half_window=10):
    # == Convolve with PSF
    ny_conv = int(convolve_half_window * 2 + 1)
    nx_conv = len(spec_oversampled_cut)
    x_t, y_t = np.meshgrid((xpos_oversampled_cut-xpos_ccd),
                           np.arange(ny_conv).astype(float) - convolve_half_window)

    # Old version
    # a0, b0, a2, b2, a1, b1, xc, yc = np.array(config_2d['trace_curvature']).astype(float)
    # dy_offset = (a2*(y_t + ypos_ccd - yc)**2 + a1*(y_t + ypos_ccd - yc) + a0) * (x_t + xpos_ccd - xc) ** 2 + (
    #         b2 * (y_t + ypos_ccd - yc) ** 2 + b1 * (y_t + ypos_ccd - yc) + b0) * (x_t + xpos_ccd - xc)
    # y_t -= dy_offset

    # using real traces
    y_center =  np.take(ypos_ccd, ypos_ccd.size//2)

    y_t += ypos_ccd - y_center

    # psf along the dispersion axis
    psf_x = np.ones(shape=(ny_conv, nx_conv), dtype=float) * focus[0, int(xpos_ccd), int(y_center)]
    # psf along the slit
    psf_y = np.ones(shape=(ny_conv, nx_conv), dtype=float) * focus[1, int(xpos_ccd), int(y_center)]
    # psf covariance
    psf_xy = np.ones(shape=(ny_conv, nx_conv), dtype=float) * focus[2, int(xpos_ccd), int(y_center)]

    return np.nansum(
        np.exp(-0.5 / (1 - psf_xy ** 2) * ((x_t / psf_x) ** 2 + (y_t / psf_y) ** 2 -
                                           2 * psf_xy * (y_t / psf_y) * (x_t / psf_x))) /
        (2 * np.pi * psf_x * psf_y * np.sqrt(1 - psf_xy ** 2)) * spec_oversampled_cut[None, :],
        axis=1)


def spec_2d_projection_parallel(spec_cur_fiber, pix_grid_input_on_ccd, focus, trace, y_pos,
                                ccd_size, ccd_gap_size, ccd_gap_left, convolve_half_window_x,
                                convolve_half_window_y):
    # Cross-disp. position of the center of the fiber
    # r = config_2d['lines_curvature'][0] * ccd_size[1]
    # dxc_offset = -(config_2d['lines_curvature'][1] - r + np.sqrt(r ** 2 - (y_pos - (ccd_size[1]*0.5)) ** 2))
    spec_res = np.zeros(shape=(int(convolve_half_window_y * 2 + 1), int(ccd_size[0] - ccd_gap_size)), dtype=float)

    # for cur_pix in range(int(dxc_offset - convolve_half_window), ccd_size[0] - ccd_gap_size):
    for cur_pix in range(ccd_size[0] - ccd_gap_size):
        pix_oversampled = np.flatnonzero((pix_grid_input_on_ccd >= (cur_pix - convolve_half_window_x)) &
                                         (pix_grid_input_on_ccd < (cur_pix + convolve_half_window_x + 1))
                                         )
        if len(pix_oversampled) > 0:
            val = spec_fragment_convolve_psf(spec_oversampled_cut=spec_cur_fiber[pix_oversampled],
                                             xpos_oversampled_cut=pix_grid_input_on_ccd[pix_oversampled],
                                             xpos_ccd=cur_pix, ypos_ccd=trace[pix_oversampled], focus=focus,
                                             convolve_half_window=convolve_half_window_y
                                             )
        else:
            val = 0
        spec_res[:, cur_pix] = val

    spec_res_projection = np.zeros(shape=(spec_res.shape[0], ccd_size[0]), dtype=float)
    spec_res_projection[:, : ccd_gap_left + 1] = spec_res[:, : ccd_gap_left + 1]
    spec_res_projection[:, ccd_gap_left + ccd_gap_size + 1:] = spec_res[:, ccd_gap_left + 1:]

    return spec_res_projection, (int(y_pos - convolve_half_window_y),
                                 int(y_pos + convolve_half_window_y))


def cosmic_rays(ccdimage, n_cr=100, std_cr=5, deep=10.0, cr_intensity=1e5):
    """
    Add traces by cosmic rays hits to CCD image
    Args:
        ccdimage: array representing the clean ccd image (spectrum)
        n_cr: approximate number of CR hits
        std_cr: standard deviation of the number of CR hits to randomize it
        deep: value defining the length of the CR hits
        cr_intensity: intensity of each hit (in counts/electrons)

    Returns: original array with added CR hits

    """
    ny, nx = ccdimage.shape
    cr_image = np.zeros_like(ccdimage)
    nc = np.abs(n_cr + int(np.random.randn(1)[0] * std_cr))
    if nc == 0:
        nc = 1
    xof = np.random.random_sample(nc) * nx
    yof = np.random.random_sample(nc) * ny
    thet = np.random.random_sample(nc) * 180. - 90.0
    phi = np.random.random_sample(nc) * 85. + 5.0
    lent = deep / np.sin(phi * np.pi / 180.0)
    for k in range(nc):
        lx = int(lent[k] * np.cos(thet[k] * np.pi / 180.0)) + 1
        x_tc = np.arange(lx) + xof[k]
        cof = yof[k] - np.tan(thet[k] * np.pi / 180.0) * xof[k]
        y_tc = np.tan(thet[k] * np.pi / 180.0) * x_tc + cof
        for i in range(0, lx):
            xt1 = np.max([0, np.min([int(x_tc[i]) - 1, nx])])
            xt2 = np.max([0, np.min([int(x_tc[i]), nx])])
            yt1 = np.max([0, np.min([int(y_tc[i]) - 1, ny])])
            yt2 = np.max([0, np.min([int(y_tc[i]), ny])])
            cr_image[xt1:xt2, yt1:yt2] = cr_intensity
    dv = 0.5
    PSF = Gaussian2DKernel(x_stddev=dv, y_stddev=dv)
    cr_image = convolve(cr_image, PSF)
    return ccdimage + cr_image


# def read_op_fib(cart, cam):
#     """
#     OBSOLETE
#     Reads config file defining the fibers mapping for each camera/channel
#     Args:
#         cart: ...
#         cam: ID of the camera (b1-b3, r1-r3, z1-z3)
#
#     Returns:
#         Relative sizes and offsets for fibers in each bundle
#
#     """
#     if (cart < 0) or (cart > 18):
#         cart = 16
#     f = open(os.path.join(DATA_DIR, 'instrument', 'fibers', config_2d['fibers_pack_name']), 'r')
#     space, bspa = (None, None)
#     for line in f:
#         if 'FIBERPARAM' in line:
#             dat = line.replace('\n', '').split('{')
#             data1 = dat[0].split(' ')
#             data1 = list(filter(None, data1))
#             if len(data1) > 2:
#                 car = int(data1[1])
#                 lap = data1[2]
#                 if (car == cart) or (cam == lap):
#                     data2 = dat[1].replace('}', '').split(' ')
#                     data2 = filter(None, data2)
#                     space = np.array([np.float(val) for val in data2])
#                     data3 = dat[2].replace('}', '').split(' ')
#                     data3 = filter(None, data3)
#                     bspa = np.array([np.float(val) for val in data3])
#                     break
#     f.close()
#     return space, bspa


def raw_data_header(h, obstime, mjd, exp_name, channel, cam, flb='science', ra=0.0, dec=0.0, airmass=1.0,
                    exp_time=900.0, bzero=32768, list_lamps='00000', gain=None, readnoise=None, gap_pos=None,
                    objname=None):
    if flb == 'science':
        flab = 'science '
    elif flb == 'arc':
        flab = 'arc     '
    elif flb == 'flat':
        flab = 'flat    '
    elif flb == 'bias':
        flab = 'bias    '

    h["FILENAME"] = (f'sdR-s-{channel}{cam}-{exp_name}.fits.gz', 'File basename')
    h["EXPOSURE"] = (int(exp_name), 'Exposure number')
    h["SPEC"] = (f'sp{cam}', 'Spectrograph name')
    h["OBSERVAT"] = ('LCO', 'Observatory')
    h["OBSTIME"] = (obstime, "Start of the observation")
    h["MJD"] = (int(mjd), 'Modified Julian Date')
    h["EXPTIME"] = (exp_time, 'Exposure time')
    h["DARKTIME"] = (exp_time, 'Dark time')
    h["IMAGETYP"] = (flab, 'Image type')
    h['INTSTART'] = (obstime, 'Start of the integration')
    h['LMST'] = (1.00, "Local mean sidereal time (approximate) [hr]")
    h['INTEND'] = ((Time(obstime)+exp_time*u.s).to_value('fits'), 'End of the integration')
    h['CCD'] = (f'{channel}{cam}', 'CCD name')
    h['CCDID'] = ('STA29925', 'Unique identifier of the CCD')
    h['CCDTYPE'] = ('STA4850 ', 'CCD type')
    h['TELESCOP'] = "SDSS 0.16m"
    h['SURVEY'] = "LVM"
    h["CCDTEMP1"] = (-104.67, "Temperature of the sensor (HEATERX 12)")
    h["CCDTEMP2"] = (-182.88, "Temperature of the sensor (HEATERX 12)")

    if objname is not None:
        h['OBJECT'] = (objname, 'Name of the target observed')
    else:
        h['OBJECT'] = 'Simulation'
    h['TILE_ID'] = (-999, "The tile_id of this observation")
    if gain is not None:
        # Note: sectors in real data go from left to rightm but from top to bottom.
        # In the simulator - from bottom to top.
        h['GAIN1'] = (gain[2], f"CCD gain AD1 [e-/ADU]")
        h['GAIN2'] = (gain[3], f"CCD gain AD1 [e-/ADU]")
        h['GAIN3'] = (gain[0], f"CCD gain AD1 [e-/ADU]")
        h['GAIN4'] = (gain[1], f"CCD gain AD1 [e-/ADU]")
        # for ind, g in enumerate(gain):
        #     h[f'GAIN{ind+1}'] = (g, f"CCD gain AD{ind+1} [e-/ADU]")
    if readnoise is not None:
        h[f'RDNOISE1'] = (readnoise[2], f'CCD read noise AD1 [e-]')
        h[f'RDNOISE2'] = (readnoise[3], f'CCD read noise AD1 [e-]')
        h[f'RDNOISE3'] = (readnoise[0], f'CCD read noise AD1 [e-]')
        h[f'RDNOISE4'] = (readnoise[1], f'CCD read noise AD1 [e-]')
        # for ind, rn in enumerate(readnoise):
        #     h[f'RDNOISE{ind+1}'] = (rn, f'CCD read noise AD{ind+1} [e-]')
    h['CCDSUM'] = ('1 1     ', 'Horizontal and vertical binning')
    h['DATASEC'] = (f"[1:{h['NAXIS1']},1:{h['NAXIS2']}]", 'Section of the detector containing data')
    h['CCDSEC'] = (f"[1:{h['NAXIS1']},1:{h['NAXIS2']}]", 'Section of the detector read out')
    # h['BIASSEC'] = (f"[{gap_pos[0]}:{gap_pos[1]},1:{h['NAXIS2']}]", 'Section of calibration / bias data')
    # h['TRIMSEC'] = (f"[1:{gap_pos[0]-1},1:{h['NAXIS2']}],[{gap_pos[1]+1}:{h['NAXIS1']},1:{h['NAXIS2']}]",
    #                 'Section with useful data')
    h['TRIMSEC1'] = (f"[1:{gap_pos[0]-1}, {gap_pos[3]+1}:{h['NAXIS2']}]", "Data section for quadrant 1")
    h['TRIMSEC2'] = (f"[{gap_pos[1]+1}:{h['NAXIS1']}, {gap_pos[3]+1}:{h['NAXIS2']}]", "Data section for quadrant 2")
    h['TRIMSEC3'] = (f"[1:{gap_pos[0] - 1}, 1:{gap_pos[2]-1}]", "Data section for quadrant 3")
    h['TRIMSEC4'] = (f"[{gap_pos[1]+1}:{h['NAXIS1']}, 1:{gap_pos[2]-1}]", "Data section for quadrant 4")
    h['BIASSEC1'] = (f"[{gap_pos[0]}:{int((gap_pos[1]+gap_pos[0])/2)},{gap_pos[3]+1}:{h['NAXIS2']}]",
                     'Overscan section for quadrant 1')
    h['BIASSEC2'] = (f"[{int((gap_pos[1] + gap_pos[0]) / 2)+1}:{gap_pos[1]},{gap_pos[3] + 1}:{h['NAXIS2']}]",
                     'Overscan section for quadrant 2')
    h['BIASSEC3'] = (f"[{gap_pos[0]}:{int((gap_pos[1]+gap_pos[0])/2)},1:{gap_pos[2]-1}]",
                     'Overscan section for quadrant 3')
    h['BIASSEC4'] = (f"[{int((gap_pos[1] + gap_pos[0]) / 2)+1}:{gap_pos[1]},1:{gap_pos[2]-1}]",
                     'Overscan section for quadrant 4')
    h["SMJD"] = (int(mjd), 'SDSS Modified Julian Date')
    h["DPOS"] = (0, "Dither position")
    h['BUFFER'] = (1, 'The buffer number read')
    h['HARTMANN'] = ('0 0     ', 'Left/right. 0=open 1=closed')

    onoff = ['OFF     ', 'ON      ']
    lamp_on_off = {"ARGON": onoff[int(list_lamps[0])],
                   "NEON": onoff[int(list_lamps[1])],
                   "LDLS": onoff[int(list_lamps[2])],
                   "HGNE": onoff[int(list_lamps[3])],
                   "XENON": onoff[int(list_lamps[3])]}
    for kv in lamp_on_off:
        h[kv] = (lamp_on_off[kv], 'Status of the corresponding lamp')
    h["BSCALE"] = 1
    h["BZERO"] = bzero

    if ra is not None:
        for t in ['Sci', 'SkyE', 'SkyW']:
            h[f"TE{t.upper()}RA"] = (ra, f'{t} telescope reported RA [deg]')
            h[f'PO{t.upper()}RA'] = (ra,  f'{t} target RA [deg]')
        h[f"TESPECRA"] = (ra, 'Spec telescope initial pointing RA [deg]')
    if dec is not None:
        for t in ['Sci', 'SkyE', 'SkyW']:
            h[f"TE{t.upper()}DE"] = (dec, f'{t} telescope reported Dec [deg]')
            h[f'PO{t.upper()}DE'] = (dec, f'{t} target Dec [deg]')
        h[f"TESPECDE"] = (dec, 'Spec telescope initial pointing Dec [deg]')
    if airmass is not None:
        for t in ['Sci', 'SkyE', 'SkyW']:
            h[f'TE{t.upper()}AM'] = (airmass, f'{t} telescope airmass')
    return h


def interp_wave(data, for_interp):

    newdata = np.zeros((data.shape[0], len(for_interp)))
    for i in range(data.shape[0]):
        newdata[i, :] = interp1d(data[i, :], np.arange(data.shape[1]), bounds_error=False,
                                fill_value='extrapolate')(for_interp)


    return newdata

def interp_trace(data, for_interp):

    newdata = np.zeros((data.shape[0], for_interp.shape[1]))
    for i in range(data.shape[0]):
        newdata[i, :] = interp1d(np.arange(data.shape[1]), data[i, :], bounds_error=False,
                                fill_value=(data[i, 0], data[i, -1]))(for_interp[i, :])


    return newdata


def cre_raw_exp(input_spectrum, fibtype, ring, position, wave_ccd, wave, trace, nfib=600, flb='s',
                channel_type="blue", cam=1, ccd_noise_factor=1.0, n_cr=130, std_cr=5,
                obstime=None, mjd=None, exp_name='0', exp_time=900.0, ra=0.0,
                dec=0.0, airmass=1.0, add_cr_hits=True, list_lamps='00000'):

    """
    Creates the Raw 2D LVM exposure from the provided spectrum
    Args:
        input_spectrum: Oversampled 2D spectrum, where each row corresponds to the spectra in one fiber
        fibtype: Array containing the information about type (science/standard/sky)
                of each fiber presented in input_spectrum
        ring: Array containing the information about the ring number for each fiber presented in input_spectrum
        position: Array containing the information about the number
                of each fiber in corresponding ring presented in input_spectrum
        wave_ccd: Wavelength grid for the output spectrum on CCD
        wave: Wavelength grid for the simulated oversampled spectrum
        ccd_noise_factor: Coefficient to be used for modification of the default CCD noise
        n_cr: Approximate number of CR hits for current exposure
        std_cr: Dispersion for n_cr to randomize it
        channel_type: Type of the current channel of spectrograph (blue, red or ir)
        cam: ID of the camera
        nfib: Total number of fibers for current camera CCD
        flb: type of the exposure
        obstime: Date and time of the observation
        mjd: Modified Julian Day of the observation
        exp_name: Number of exposure
        exp_time: exposure time
        ra: RA of the pointing
        dec: DEC of the pointing
        add_cr_hits: add or not CR hits
        list_lamps: State of the lamps (argon/neon/flat/HgNe/Xenon; 1=enabled, 0=disabled)

    Returns: Tuple of 2 HDUs containing projected spectrum+header and associated bias+header

    """
    if channel_type not in ['blue', 'red', 'ir']:
        log.error(f"Unrecognized type of the spectral channel: {channel_type}")
        return None

    try:
        ccd_props = ascii.read(os.path.join(DATA_DIR, 'instrument', f"{config_2d['ccd_properties']}"))
    except FileNotFoundError:
        log.error(f"File defining the CCD properties is not found! 2D simulations cannot be performed further")
        return None
    ccd_props = ccd_props[ccd_props['channel'] == channel_type][0]
    channel_index = {'blue': 'b', 'red': 'r', 'ir': 'z'}
    ccd_size = [ccd_props['nx'], ccd_props['ny']]
    ccd_gap_size = ccd_props['x2'] - ccd_props['x1'] - 1
    ccd_middle_x_pos = int((ccd_props['x1'] + ccd_props['x2']) / 2.)
    cam = str(int(cam))
    output = np.zeros(shape=(ccd_size[1], ccd_size[0]))

    if flb != 'bias':

        # fibs1, bunds1 = read_op_fib(1, channel_index[channel_type] + cam)
        try:
            # TODO: at the moment, these files are of 4120x4080 size. Perhaps they should either take into account the
            #  gap, or be of 4080x4080 size. For now, I cut the excess
            focus = fits.getdata(os.path.join(DATA_DIR, 'focus',
                                              f"{config_2d['psf_rootname']}_{channel_type}{cam}.fits.gz"),
                                 0, header=False).T
        except FileNotFoundError:
            focus = np.ones([3, ccd_size[0]-ccd_gap_size, ccd_size[1]], dtype=float)
            focus[1, :, :] = 0.9
            focus[2, :, :] = 0.0
            log.warning(f'PSF data for {channel_type} channel is not found. Using default PSF = 1 pixel')
        # Fiber mapping
        # TODO: perhaps this is unnecessary - parameters ring, fibtype, position,...
        #  were derived from full_array.dat, ...,
        #  and these are very similar to plugmap.dat. Maybe these two files should be merged into a single file?
        fibers_mapping = ascii.read(os.path.join(DATA_DIR, 'instrument', 'fibers',
                                                 f"{config_2d['fibers_ccd_map_name']}"))
        fib_id_on_slit = np.zeros(len(ring), dtype=int) - 1
        y_pos = np.zeros(len(ring), dtype=float) - 1
        for i in range(len(ring)):
            cur_fiber_num_in_mapping = np.flatnonzero((fibers_mapping['ring_id'] == ring[i]) &
                                                      (fibers_mapping['type'] == fibtype[i]) &
                                                      (fibers_mapping['slit'] == 'slit' + cam) &
                                                      (fibers_mapping['fiber_id'] == position[i]))
            if len(cur_fiber_num_in_mapping) > 0:
                fib_id_on_slit[i] = np.atleast_1d(fibers_mapping['id'][cur_fiber_num_in_mapping])[0]
                y_pos[i] = np.atleast_1d(fibers_mapping[f'y_{channel_type}'][cur_fiber_num_in_mapping])[0]


        fib_id_in_ring = np.zeros(nfib, dtype=int) - 1
        current_y_pos = np.zeros(nfib, dtype=int) - 1
        # This array has the IDs = -1 for those fibers that are not used in the simulations
        for cur_fiber_num in range(nfib):
            mask = (fib_id_on_slit == cur_fiber_num)
            nt = np.flatnonzero(mask)
            if len(nt) > 0:
                fib_id_in_ring[cur_fiber_num] = np.atleast_1d(nt)[0]
                current_y_pos[cur_fiber_num] = y_pos[mask][0]

        # Wavelength solution
        # TODO: This should be defined for each fiber to account for the differences in wavelength solution between them
        # Done!!!
        pix_grid_input_on_ccd = interp_wave(wave_ccd, wave)
        trace = interp_trace(trace, pix_grid_input_on_ccd)

        log.info(f"Project the spectra of camera #{cam} and {channel_type} channel onto CCD")
        # Half size of the window for convolution
        convolve_half_window_x = np.ceil(np.nanmax(focus[0, :, :])*6).astype(int)
        # Value is higher to get the curvature into account
        convolve_half_window_y = np.ceil(np.nanmax(focus[1, :, :])*6*1.5).astype(int)
        with tqdm_joblib(tqdm(total=np.sum(fib_id_in_ring >= 0))):
            results = Parallel(n_jobs=n_process)(delayed(spec_2d_projection_parallel)(
                input_spectrum[fib_id_in_ring[cur_fiber_num], :],
                pix_grid_input_on_ccd[cur_fiber_num, :], focus, trace[cur_fiber_num],
                current_y_pos[cur_fiber_num], ccd_size,
                ccd_gap_size, ccd_props['x1'], convolve_half_window_x, convolve_half_window_y)
                                                 for cur_fiber_num in range(nfib) if fib_id_in_ring[cur_fiber_num] >= 0)

        for res_element in results:
            output[res_element[1][0]: res_element[1][1]+1, :] += res_element[0]

    sig = ccd_noise_factor * ccd_props['noise']
    sector_map = {'1': ['x0', 'x1', 'y0', 'y1'],
                  '2': ['x2', 'x3', 'y0', 'y1'],
                  '3': ['x0', 'x1', 'y2', 'y3'],
                  '4': ['x2', 'x3', 'y2', 'y3']}

    for key in sector_map.keys():
        y0 = ccd_props[sector_map[key][2]]
        y1 = ccd_props[sector_map[key][3]] + 1
        x0 = np.min([ccd_props[sector_map[key][0]], ccd_middle_x_pos])
        x1 = np.max([ccd_props[sector_map[key][1]] + 1, ccd_middle_x_pos])

        output[y0: y1, x0: x1] = \
            output[y0: y1, x0: x1] / ccd_props[f'gain_{key}'] + \
            np.random.randn(y1 - y0, x1 - x0) * sig / ccd_props[f'gain_{key}'] + \
            ccd_props['bias'] + ccd_props[f'bias_add_{key}']

    if add_cr_hits:
        output = cosmic_rays(output, n_cr=n_cr, std_cr=std_cr)

    if channel_type in ['blue', 'ir']:
        output = np.fliplr(output)

    output_hdus = (ccdspec_to_hdu(output, obstime, mjd, exp_name, channel_index[channel_type], cam,
                                  flb=flb, exp_time=exp_time, ra=ra.value, dec=dec.value, bzero=32768,
                                  ccd_props=ccd_props, list_lamps=list_lamps, airmass=airmass))
    return output_hdus


def ccdspec_to_hdu(data, obstime, mjd, exp_name, channel=None, cam=None, flb='science', exp_time=0., ra=None, dec=None,
                   airmass=1.0, bzero=32768, ccd_props=None, list_lamps='00000'):
    """
    Saturate the output array and convert it to the fits format with a proper fits header
    Args:
        data: numpy array containing the reprojected spectrum (or bias)
        obstime: Time of observations
        mjd: Modified Julian Day of the observation
        exp_name: name of exposure
        channel: index corresponding to the channel of spectrograph (b, r, or z)
        cam: index of camera
        flb: type of exposure
        exp_time: Exposure time
        ra: RA of the pointing
        dec: Dec of the pointing
        airmass: airmass assumed in simulations
        bzero: bzero level in fits file (corresponds to saturation level + 1)
        ccd_props: row defining the properties of CCD for current channel
        list_lamps: State of the lamps (argon/neon/flat/HgNe/Xenon; 1=enabled, 0=disabled)

    Returns: HDU list where primary HDU contains this data

    """
    data -= bzero
    data[data > (bzero - 1)] = float(bzero - 1)
    data = np.round(data).astype('int16')
    hdu = fits.PrimaryHDU(data)
    hdu.header["NAXIS"] = 2
    hdu.header["NAXIS1"] = data.shape[1]
    hdu.header["NAXIS2"] = data.shape[0]
    if ccd_props is not None:
        gain = [ccd_props['gain_1'], ccd_props['gain_2'], ccd_props['gain_3'], ccd_props['gain_3']]
        readnoise = [ccd_props['noise']]*4
        gap_pos = [ccd_props['x1']+2, ccd_props['x2'], ccd_props['y1']+2, ccd_props['y2']]  # added +1 to convert to fits coordinates (starts from 1)
    else:
        gain = [1, 1, 1, 1]
        readnoise = [3.8]*4
        gap_pos = [2040, 2080, 2040, 2080]
    hdu.header = raw_data_header(hdu.header, obstime, mjd, exp_name, channel, cam, flb=flb,
                                 exp_time=exp_time, ra=ra, dec=dec, bzero=bzero, gain=gain,
                                 readnoise=readnoise, gap_pos=gap_pos, list_lamps=list_lamps,
                                 airmass=airmass)
    hlist = fits.HDUList([hdu])
    hlist.update_extend()
    return hlist

