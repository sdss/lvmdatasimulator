# encoding: utf-8
# @Author: Oleg Egorov, Enrico Congiu, Hector Ibarra
# @Date: Oct 28, 2022
# @Filename: projection2d.py
# @License: BSD 3-Clause
# @Copyright: Oleg Egorov, Enrico Congiu, Hector Ibarra

import os.path
import numpy as np
from lvmdatasimulator import COMMON_SETUP_2D as config_2d
from lvmdatasimulator import DATA_DIR, n_process
from astropy.io import fits, ascii
from lvmdatasimulator import log
from scipy.interpolate import interp1d
from astropy.convolution import convolve
from astropy.convolution.kernels import Gaussian2DKernel
from multiprocessing import Pool
from tqdm import tqdm


def spec_fragment_convolve_psf(spec_oversampled_cut=None, xpos_oversampled_cut=None, xpos_ccd=None, ypos_ccd=None,
                               focus=None, convolve_half_window=10):
    # == Convolve with PSF
    ny_conv = int(convolve_half_window*2+1)
    nx_conv = len(spec_oversampled_cut)
    x_t, y_t = np.meshgrid((xpos_oversampled_cut-xpos_ccd),
                           np.arange(ny_conv).astype(float) - convolve_half_window)

    a0, b0, a2, b2, a1, b1, xc, yc = np.array(config_2d['trace_curvature']).astype(float)
    dy_offset = (a2*(y_t + ypos_ccd - yc)**2 + a1*(y_t + ypos_ccd - yc) + a0) * (x_t + xpos_ccd - xc) ** 2 + (
            b2 * (y_t + ypos_ccd - yc) ** 2 + b1 * (y_t + ypos_ccd - yc) + b0) * (x_t + xpos_ccd - xc)
    y_t -= dy_offset
    # TODO check what is in the FOCUS files. What in the 1st and 2nd column? PSF along the slit and disp?
    #  Or vice-versa? For now - assume that 0 is along dispersion axis (in agreement with the code below),
    #  although it is cotrary to Hector's example
    psf_x = np.ones(shape=(ny_conv, nx_conv), dtype=float) * focus[0, int(xpos_ccd), int(ypos_ccd)]  # psf along the dispersion axis
    psf_y = np.ones(shape=(ny_conv, nx_conv), dtype=float) * focus[1, int(xpos_ccd), int(ypos_ccd)]  # psf along the slit
    psf_xy = np.ones(shape=(ny_conv, nx_conv), dtype=float) * focus[2, int(xpos_ccd), int(ypos_ccd)]  # psf covariance

    return np.nansum(
        np.exp(-0.5 / (1 - psf_xy ** 2) * ((x_t / psf_x) ** 2 + (y_t / psf_y) ** 2 -
                                           2 * psf_xy * (y_t / psf_y) * (x_t / psf_x))) /
        (2 * np.pi * psf_x * psf_y * np.sqrt(1 - psf_xy ** 2)) * spec_oversampled_cut[None, :],
        axis=1)


def spec_2d_projection_parallel(params):
    spec_cur_fiber, pix_grid_input_on_ccd, focus, y_pos, \
        ccd_size, ccd_gap_size, ccd_gap_left = params

    # Cross-disp. position of the center of the fiber

    r = config_2d['lines_curvature'][0]*ccd_size[1]
    dxc_offset = -(config_2d['lines_curvature'][1] - r + np.sqrt(r ** 2 - (y_pos - (ccd_size[1]*0.5)) ** 2))
    convolve_half_window_x = 8  # Half size of the window for convolution
    convolve_half_window_y = 15  # Value is higher to get the curvature into account
    spec_res = np.zeros(shape=(int(convolve_half_window_y * 2 + 1), int(ccd_size[0] - ccd_gap_size)), dtype=float)

    # for cur_pix in range(int(dxc_offset - convolve_half_window), ccd_size[0] - ccd_gap_size):
    for cur_pix in range(ccd_size[0] - ccd_gap_size):
        pix_oversampled = np.flatnonzero((pix_grid_input_on_ccd >= (cur_pix - dxc_offset - convolve_half_window_x)) &
                                         (pix_grid_input_on_ccd < (cur_pix - dxc_offset + convolve_half_window_x + 1)))
        if len(pix_oversampled) > 0:
            val = spec_fragment_convolve_psf(spec_oversampled_cut=spec_cur_fiber[pix_oversampled],
                                             xpos_oversampled_cut=pix_grid_input_on_ccd[pix_oversampled]+dxc_offset,
                                             xpos_ccd=cur_pix, ypos_ccd=y_pos, focus=focus,
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
    nc = np.abs(n_cr + np.int(np.random.randn(1)[0] * std_cr))
    if nc == 0:
        nc = 1
    xof = np.random.random_sample(nc) * nx
    yof = np.random.random_sample(nc) * ny
    thet = np.random.random_sample(nc) * 180. - 90.0
    phi = np.random.random_sample(nc) * 85. + 5.0
    lent = deep / np.sin(phi * np.pi / 180.0)
    for k in range(nc):
        lx = np.int(lent[k] * np.cos(thet[k] * np.pi / 180.0)) + 1
        x_tc = np.arange(lx) + xof[k]
        cof = yof[k] - np.tan(thet[k] * np.pi / 180.0) * xof[k]
        y_tc = np.tan(thet[k] * np.pi / 180.0) * x_tc + cof
        for i in range(0, lx):
            xt1 = np.max([0, np.min([np.int(x_tc[i]) - 1, nx])])
            xt2 = np.max([0, np.min([np.int(x_tc[i]), nx])])
            yt1 = np.max([0, np.min([np.int(y_tc[i]) - 1, ny])])
            yt2 = np.max([0, np.min([np.int(y_tc[i]), ny])])
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


def raw_data_header(h, field_name, mjd, exp_name, typ, flb='science', ra=0.0, dec=0.0, azim=180.0, alt=90.0,
                    exp_time=900.0, bzero=32768):
    if flb == 'science':
        flab = 'science '
    elif flb == 'arc':
        flab = 'arc     '
        # expt = 4.0
    elif flb == 'flat':
        flab = 'flat    '
        # expt = 25.0
    elif flb == 'bias':
        flab = 'bias    '
        expt = 0

    h["TELESCOP"] = 'LVM data simulator'
    h["FILENAME"] = 'sdR-' + typ + '-' + f"{exp_name:08}" + '.fits'
    h["CAMERAS"] = typ + '      '
    h["EXPOSURE"] = exp_name
    h["FLAVOR"] = (flab, 'exposure type, SDSS spectro style')
    h["MJD"] = (np.int(mjd), 'APO fMJD day at start of exposure')
    h["TAI-BEG"] = ((np.float(mjd) + 0.25) * 24.0 * 3600.0, 'MJD(TAI) seconds at start of integration')
    h["DATE-OBS"] = ('2012-03-20T06:00:00', 'TAI date at start of integration')
    h["NAME"] = (field_name + '-' + mjd + '-01', 'The name of the currently loaded plate')
    h["CONFIID"] = (field_name, 'The currently FPS configuration')
    h["OBJSYS"] = ('ICRS    ', 'The TCC objSys')
    if ra is not None:
        h["RA"] = (ra, 'RA of telescope boresight (deg)')
        h["RADEG"] = (ra , 'RA of telescope pointing(deg)')
    if dec is not None:
        h["DEC"] = (dec, 'Dec of telescope boresight (deg)')
        h["DECDEG"] = (dec, 'Dec of telescope pointing (deg)')
    if azim is not None:
        h["AZ"] = (azim, 'Azimuth axis pos. (approx, deg)')
    if alt is not None:
        h["ALT"] = (alt, 'Altitude axis pos. (approx, deg)')
    if 'flat    ' in flab:
        h["FF"] = ('1 1 1 1 ', 'FF lamps 1:on 0:0ff')
        h["NE"] = ('0 0 0 0 ', 'NE lamps 1:on 0:0ff')
        h["HGCD"] = ('0 0 0 0 ', 'HGCD lamps 1:on 0:0ff')
        h["FFS"] = ('1 1 1 1 1 1 1 1', 'Flatfield Screen 1:closed 0:open')
    elif 'science ' in flab:
        h["FF"] = ('0 0 0 0 ', 'FF lamps 1:on 0:0ff')
        h["NE"] = ('0 0 0 0 ', 'NE lamps 1:on 0:0ff')
        h["HGCD"] = ('0 0 0 0 ', 'HGCD lamps 1:on 0:0ff')
        h["FFS"] = ('0 0 0 0 0 0 0 0', 'Flatfield Screen 1:closed 0:open')
    elif 'arc     ' in flab:
        h["FF"] = ('0 0 0 0 ', 'FF lamps 1:on 0:0ff')
        h["NE"] = ('1 1 1 1 ', 'NE lamps 1:on 0:0ff')
        h["HGCD"] = ('1 1 1 1 ', 'HGCD lamps 1:on 0:0ff')
        h["FFS"] = ('1 1 1 1 1 1 1 1', 'Flatfield Screen 1:closed 0:open')
    h["REQTIME"] = (exp_time, 'requested exposure time')
    h["EXPTIME"] = (exp_time, 'measured exposure time, s')
    h["BSCALE"] = 1
    h["BZERO"] = bzero
    return h


def cre_raw_exp(input_spectrum, fibtype, ring, position, wave_ccd, wave, nfib=600, flb='s',
                channel_type="blue", cam=1, ccd_noise_factor=1.0, n_cr=130, std_cr=5,
                mjd='45223', field_name='00000', exp_name='0', exp_time=900.0, ra=0.0,
                dec=0.0, add_cr_hits=True):

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
        flb: types of the exposures (???)
        mjd: Date of the observation (???)
        field_name: Name of the observed field (plate) (???)
        exp_name: Name of exposure (???)
        exp_time: exposure time
        ra: RA of the pointing
        dec: DEC of the pointing
        add_cr_hits: add or not CR hits

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
                                              f"{config_2d['psf_rootname']}_{channel_type}1.fits.gz"),
                                 0, header=False).T[:, :ccd_size[0]-ccd_gap_size, :]
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
        pix_grid_input_on_ccd = interp1d(wave_ccd, np.arange(len(wave_ccd)), bounds_error=False,
                                         fill_value='extrapolate')(wave)

        log.info(f"Project the spectra of camera #{cam} and {channel_type} channel onto CCD")
        # results = []
        # for cur_fiber_num in range(nfib):
        #     if fib_id_in_ring[cur_fiber_num]<0:
        #         continue
        #     results.append(spec_2d_projection_parallel((input_spectrum[fib_id_in_ring[cur_fiber_num], :],
        #                                    pix_grid_input_on_ccd, cur_fiber_num, nfib, bunds1,
        #                                    fibs1, focus,
        #                                    ccd_size, ccd_gap_size, ccd_props['x1'])))


        # pixtab_wl_solution = apply_wl_solution(pix_grid_input_on_ccd)

        # results = []
        # for cur_fiber_num in range(nfib):
        #     if fib_id_in_ring[cur_fiber_num] >= 0:
        #         results.append(spec_2d_projection_parallel((input_spectrum[fib_id_in_ring[cur_fiber_num], :],
        #                                                               pix_grid_input_on_ccd, cur_fiber_num, nfib, bunds1,
        #                                                               fibs1, focus,
        #                                                               ccd_size, ccd_gap_size, ccd_props['x1'])))

        with Pool(n_process) as p:
            results = list(tqdm(p.imap(spec_2d_projection_parallel, [(input_spectrum[fib_id_in_ring[cur_fiber_num], :],
                                                                      pix_grid_input_on_ccd, focus,
                                                                      current_y_pos[cur_fiber_num], ccd_size,
                                                                      ccd_gap_size, ccd_props['x1'])
                                                                     for cur_fiber_num in range(nfib)
                                                                     if fib_id_in_ring[cur_fiber_num] >= 0]),
                                total=np.sum(fib_id_in_ring >= 0)))
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

    output_hdus = (ccdspec_to_hdu(output, field_name, mjd, exp_name, channel_index[channel_type],
                                  flb=flb, exp_time=exp_time, ra=ra.value, dec=dec.value, bzero=32768))
    return output_hdus


def ccdspec_to_hdu(data, field_name, mjd, exp_name, channel=None, flb='science', exp_time=0., ra=None, dec=None,
                   bzero=32768):
    """
    Saturate the output array and convert it to the fits format with a proper fits header
    Args:
        data: numpy array containing the reprojected spectrum (or bias)
        field_name: Name of the observed field (plate)
        mjd: Date of the observations (???)
        exp_name: name of exposure (???)
        channel: index corresponding to the channel of spectrograph (b, r, or z)
        flb: type of exposure
        exp_time: Exposure time
        ra: RA of the pointing
        dec: Dec of the pointing
        bzero: bzero level in fits file (corresponds to saturation level + 1)

    Returns: HDU list where primary HDU contains this data

    """
    data -= bzero
    data[data > (bzero - 1)] = float(bzero - 1)
    data = np.round(data).astype('int16')
    hdu = fits.PrimaryHDU(data)
    hdu.header["NAXIS"] = 2
    hdu.header["NAXIS1"] = data.shape[1]
    hdu.header["NAXIS2"] = data.shape[0]
    hdu.header = raw_data_header(hdu.header, field_name, mjd, exp_name, channel, flb=flb,
                                 exp_time=exp_time, ra=ra, dec=dec, bzero=bzero)
    hlist = fits.HDUList([hdu])
    hlist.update_extend()
    return hlist

