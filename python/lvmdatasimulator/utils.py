# encoding: utf-8
#
# @Author: Oleg Egorov, Enrico Congiu
# @Date: Nov 12, 2021
# @Filename: field.py
# @License: BSD 3-Clause
# @Copyright: Oleg Egorov, Enrico Congiu
import os.path
import numpy as np
import re
import pyCloudy as pc
import astropy.units as u
import lvmdatasimulator

from dataclasses import dataclass
from astropy.io import fits, ascii
from astropy.table import Table
from astropy.units import UnitConversionError
from astropy.convolution import convolve_fft
from shapely.geometry import Point, Polygon, box
from pyneb import RedCorr
from lvmdatasimulator import log
from sympy import divisors
from scipy.interpolate import RectBivariateSpline, interp1d
from spectres import spectres
from astropy.convolution import Gaussian1DKernel, convolve
from astropy.io.misc import yaml
import contextlib
import joblib

from lvmdatasimulator import DATA_DIR


# unit conversions
r_to_erg_ha = 5.661e-18 * u.erg/(u.cm * u.cm * u.s * u.arcsec**2)

@dataclass(frozen=True)
class Constants:
    h: u.erg * u.s = 6.6260755e-27 * u.erg * u.s  # Planck's constant in [erg*s]
    c: u.AA * u.s = 2.99792458e18 * u.AA / u.s  # Speed of light in [A/s]


@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into tqdm progress bar given as argument (taken from stackoverflow)"""
    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()


def assign_units(my_object, variables, default_units):
    """
    Convert every involved variables to the astropy.units assuming the default prescriptions and equivalencies
    :param my_object : Object to check the properties
    :param variables: list of names of properties to check. If my_object is None, then it is the list of variables
    :param default_units: units to assign for each variable/property
    :return: object with the updated properties, or the updated list with variables
    """
    out_list = []
    for ind, curvar in enumerate(variables):
        if my_object is None and curvar is None:
            out_list.append(None)
            continue
        elif my_object is not None and my_object.__getattribute__(curvar) is None:
            continue
        try:
            if my_object is None:
                out_list.append(curvar << default_units[ind])
            else:
                my_object.__setattr__(curvar, (my_object.__getattribute__(curvar) << default_units[ind]))
        except UnitConversionError:
            if my_object is not None:
                add_str = f' of {my_object.__class__.__name__}'
            else:
                add_str = ""
            log.error(f"Wrong unit for parameter {curvar}{add_str} ({my_object.__getattribute__(curvar).unit}, "
                      f"but should be {default_units[ind]})")
    if my_object is None:
        return out_list
    else:
        return my_object


@dataclass
class Chunk:

    data: np.array
    original_position: tuple
    overlap: tuple

    def __post_init__(self):
        self.shape = self.data.shape

    def set_data(self, newdata, resize=True, orig_shape=None):

        y0 = 0
        y1 = newdata.shape[1]
        x0 = 0
        x1 = newdata.shape[2]
        if self.original_position[0][0] != 0: y0 += self.overlap
        if self.original_position[1][0] != 0: x0 += self.overlap
        if self.original_position[0][1] < orig_shape[1]: y1 -= self.overlap
        if self.original_position[1][1] < orig_shape[2]: x1 -= self.overlap

        self.data = newdata[:, y0: y1, x0: x1]

    def __str__(self):
        return f'Original position: {self.original_position},\nOverlap: {self.overlap}'

    def mean(self):
        return np.nanmean(self.data)


def round_up_to_odd(f):
    try:
        return np.ceil(f) // 2 * 2 + 1
    except UnitConversionError:
        unit = f.unit
        return (np.ceil(f.value) // 2 * 2 + 1) * unit


def check_overlap(hdu, out_shape):
    """
    Checks if the current nebula (defined in hdu) is overlapping with the FOV
    :param hdu: HDU defining the current nebula brightness distribution
    :param out_shape: tuple (height, width) defining the size of the FOV in pixels
    :return: True if the nebula is overlapping with the FOV; otherwise is False
    """

    if ((hdu.header['X0'] + hdu.header['NAXIS1']) < 0) or (
            (hdu.header['Y0'] + hdu.header['NAXIS2']) < 0) or (
            hdu.header['Y0'] >= out_shape[0]) or (hdu.header['X0'] >= out_shape[1]):
        return False
    return True


def set_default_dict_values(mydict, key_to_check, default_value, unit=None):
    """
    Checks the dictionary for the keyword and set the default values if it is missing
    :param mydict: dictionary to be checked
    :param key_to_check: keyword to check
    :param default_value: default value to be used if keyword is missing
    :param unit: astropy.unit to be use for considered value
    :return: modified dictionary
    """
    if key_to_check not in mydict:
        mydict[key_to_check] = default_value
    if unit is not None and not isinstance(mydict[key_to_check], str) and mydict[key_to_check] is not None:
        mydict[key_to_check] = (mydict[key_to_check] << unit)


def calc_circular_mask(radius, center=None, size=None):
    """
    Calculate the exact circular mask accounting for the fractions of the pixels at the edge
    :param radius: exact radius of the circular mask (in pixels)
    :param center: position of the center of the mask (in pixels; default = in the center of the circle)
    :param size: size of the mask (in pixels; default = 2*radius+1)
    :return: created mask (numpy.array)
    """
    if size is None:
        size = np.ceil(2 * radius)
    size = size.astype(int)
    if size % 2 == 0:
        size += 1

    if center is None:
        center = ((size - 1) / 2, (size - 1) / 2)

    xx, yy = np.meshgrid(np.arange(size), np.arange(size))
    mask = np.zeros(shape=(size, size), dtype=np.float32)

    rec_inside = np.ones_like(mask, dtype=bool).ravel()
    rec_outside = np.ones_like(mask, dtype=bool).ravel()
    for offsets in [(-0.5, -0.5), (-0.5, 0.5), (0.5, -0.5), (0.5, 0.5)]:
        rec_inside = rec_inside & (((xx.ravel()+offsets[0]) - center[0]) ** 2 +
                                   ((yy.ravel()+offsets[1]) - center[0]) ** 2 <= (radius ** 2))
        rec_outside = rec_outside & (((xx.ravel() + offsets[0]) - center[0]) ** 2 +
                                     ((yy.ravel() + offsets[1]) - center[0]) ** 2 > (radius ** 2))
    mask.ravel()[rec_inside] = 1.
    indexes_edge = np.flatnonzero((~rec_inside) & (~rec_outside))

    circle = Point(center[0], center[1]).buffer(radius)
    for index in indexes_edge:
        rect = box(minx=xx.ravel()[index]-0.5, miny=yy.ravel()[index]-0.5,
                   maxx=xx.ravel()[index]+0.5, maxy=yy.ravel()[index]+0.5)
        mask.ravel()[index] = circle.intersection(rect).area
    return mask


def ism_extinction(av=None, wavelength=None, r_v=3.1, ext_law='F99'):
    """
    Compute the corrections of the fluxes at different wavelength for ISM extinction
    :param av: array (1D or 2D) with the A_V at each position
    :param wavelength: 1D array or float with the wavelength grid
    :param r_v: R_V = AV/E_BV. Default value is 3.1
    :param ext_law: one of the extinction laws defined in pyneb (default='F99')
    :return: array with the coefficients to be used for correction of the 'real' fluxes
    """
    rc = RedCorr(E_BV=av/r_v, R_V=r_v, law=ext_law)
    return 1. / rc.getCorr(wavelength)


def convolve_array(to_convolve, kernel, selected_points_x, selected_points_y,
                   allow_huge=True, normalize_kernel=False, pix_size=1, nchunks=-1):

    # deciding how much to divide
    orig_shape = to_convolve.shape
    orig_shape_arcsec = orig_shape[1:] * pix_size

    if nchunks == -1:
        if max(orig_shape_arcsec) > 1700:
            nchunks = 25
        elif max(orig_shape_arcsec) > 1200:
            nchunks = 16
        elif max(orig_shape_arcsec) > 660:
            nchunks = 4
        else:
            nchunks = 1

    # defining the overlap as the size of the kernel + some room
    overlap = 1.1 * np.max(kernel.shape)

    if nchunks > 1:
        log.info(f'Dividing the array in {nchunks} with an overlap of {overlap*pix_size} arcsec')
        # dividing the cube in chuncks before convolving
        chunks = chunksize(to_convolve, nchunks=nchunks, overlap=overlap)
        for i, chunk in enumerate(chunks):

            tmp = convolve_fft(chunk.data, kernel, allow_huge=allow_huge,
                               normalize_kernel=normalize_kernel)
            chunk.set_data(tmp.astype(np.float32), resize=True, orig_shape=orig_shape)

        convolved = reconstruct_cube(chunks, orig_shape)

    else:
        # convolving the cube in a single try
        convolved = convolve_fft(to_convolve, kernel, allow_huge=allow_huge,
                                 normalize_kernel=normalize_kernel)

    # data_in_aperture = convolved[:, np.round(selected_points_y).astype(int), np.round(selected_points_x).astype(int)]
    data_in_aperture = np.zeros(shape=(convolved.shape[0], len(selected_points_x)),
                                dtype=np.float32)
    xx, yy = (np.arange(convolved.shape[2]), np.arange(convolved.shape[1]))

    for ind, conv_slice in enumerate(convolved):
        p = RectBivariateSpline(xx, yy, conv_slice.T, kx=1, ky=1)
        data_in_aperture[ind, :] = p.ev(selected_points_x, selected_points_y)

    return data_in_aperture


def chunksize(cube, nchunks=4, overlap=40):

    overlap = int(overlap)

    if nchunks == np.sqrt(nchunks) ** 2:
        max_chunks = int(np.sqrt(nchunks))
        min_chunks = int(np.sqrt(nchunks))
    else:
        divs = divisors(nchunks)
        id_max = len(divs) // 2

        # checking how much each dimension should be divided
        max_chunks = divs[id_max]
        min_chunks = divs[id_max - 1]

    original_shape = cube.shape[1: ]  # saving the size of the cube in the y and x dimension
    if original_shape[0] == original_shape[1]:
        min_dim_id = 0
        max_dim_id = 1
    else:
        min_dim_id = np.argmin(original_shape)
        max_dim_id = np.argmax(original_shape)

    # define chunk size
    size_min_dim = int(original_shape[min_dim_id] / min_chunks)
    size_max_dim = int(original_shape[max_dim_id] / max_chunks)

    # define the corners of the chunks without overlap
    cor_min_dim = np.zeros(min_chunks+1, dtype=int)
    cor_max_dim = np.zeros(max_chunks+1, dtype=int)

    for i in range(min_chunks):
        cor_min_dim[i] = i * size_min_dim
    for i in range(max_chunks):
        cor_max_dim[i] = i * size_max_dim

    cor_min_dim[-1] = original_shape[min_dim_id]
    cor_max_dim[-1] = original_shape[max_dim_id]

    original_corners = []
    for i in range(min_chunks):
        for j in range(max_chunks):
            if min_dim_id == 0:
                corners = ((cor_min_dim[i], cor_min_dim[i+1]), (cor_max_dim[j], cor_max_dim[j+1]))
            else:
                corners = ((cor_max_dim[j], cor_max_dim[j+1]), (cor_min_dim[i], cor_min_dim[i+1]))
            original_corners.append(corners)

    assert len(original_corners) == nchunks, f'{len(original_corners)} but {nchunks} chunks'

    chunk_list = []

    for corner in original_corners:

        # defining the new corners
        y0 = int(corner[0][0])
        y1 = int(corner[0][1])
        x0 = int(corner[1][0])
        x1 = int(corner[1][1])

        if y0 != 0: y0 -= overlap
        if x0 != 0: x0 -= overlap
        if y1 < original_shape[0]: y1 += overlap
        if x1 < original_shape[1]: x1 += overlap

        tmp_chunk = Chunk(cube[:, y0: y1, x0: x1],   # this is the actual array
                          corner,
                          overlap)

        chunk_list.append(tmp_chunk)

    return chunk_list


def reconstruct_cube(chunks, orig_shape):

    # create the empty cube

    new_cube = np.zeros(orig_shape, dtype=np.float32)

    # loop through the chunks
    for chunk in chunks:
        corner = chunk.original_position
        data = chunk.data
        # shape = chunk.shape

        # resizing moved to chunk to reduce memory usage
        # y0 = 0
        # y1 = shape[1]
        # x0 = 0
        # x1 = shape[2]

        # if corner[0][0] != 0: y0 += chunk.overlap
        # if corner[1][0] != 0: x0 += chunk.overlap
        # if corner[0][1] < orig_shape[1]: y1 -= chunk.overlap
        # if corner[1][1] < orig_shape[2]: x1 -= chunk.overlap

        new_cube[:, corner[0][0]: corner[0][1], corner[1][0]: corner[1][1]] = data

    return new_cube


def models_grid_summary(model_type='cloudy'):
    """
    Return the summary of the models grid (Cloudy or continuum [SB99])
    :param model_type: "cloudy" or "continuum"
    :return: astropy.table with the summary
    """
    if model_type.lower() == 'cloudy':
        name_models = lvmdatasimulator.CLOUDY_MODELS
    elif model_type.lower() in ['continuum', 'contin', 'cont']:
        name_models = lvmdatasimulator.CONTINUUM_MODELS
    elif model_type.lower() == 'mappings':
        name_models = lvmdatasimulator.MAPPINGS_MODELS
    else:
        print("Incorrect model_type.")
        return None
    with fits.open(name_models) as hdu:
        return Table(hdu['Summary'].data)


def save_cloudy_models(path_to_models, models_rootname, fileout=None):
    """
    Read the output of Cloudy and save them to the file adapted for the lvmdatasimulator
    """
    if fileout is None:
        print("Saving the Clody models: fileout is required")
        return
    if os.path.isdir(path_to_models):
        print(f"No such directory: {path_to_models}")
        return
    Ms = pc.load_models(os.path.join(path_to_models, models_rootname), read_grains=False)
    out_tab = Table(names=["Model_ID", "Geometry", 'Z', "qH", 'LogLsun', "Teff", "nH", "Rin", "Rout",
                           "Nzones", "Nlines", "Distance", "Flux_Ha", "Source_model"],
                    dtype=[str, str, float, float, float, float, float, float, float, int, int, float, float, str])
    hdul = fits.HDUList(
        [fits.PrimaryHDU()]
    )
    mod_id_shell = 0
    mod_id_cloud = 0
    for mod_id, M in enumerate(Ms):
        r_in = np.round(M.r_in/3.08567758e18, 2)
        r_out = np.round(M.r_out/3.08567758e18, 2)
        if (r_out-r_in) <= 0.05:
            continue
        dens = np.round(M.nH_mean, 1)
        qH = np.round(np.log10(M.Q0), 1)
        loglum = re.split(r"l(\d.\d)_", M.model_name_s)
        if len(loglum) > 1:
            loglum = float(loglum[1])
        else:
            loglum = np.nan
        OH = np.round(10 ** (12 + M.abund['O'] - 8.69), 2)
        Teff = M.Teff
        radius = (M.radius / 3.08567758e18 - r_in) / (r_out - r_in)

        hdul.append(fits.ImageHDU())
        if r_in > 1:
            mod_id_shell += 1
            hdul[-1].header['Model_ID'] = f"Shell_{mod_id_shell}"
            hdul[-1].header['Geometry'] = 'Shell'
        else:
            mod_id_cloud+=1
            hdul[-1].header['Model_ID'] = f"Cloud_{mod_id_cloud}"
            hdul[-1].header['Geometry'] = 'Cloud'
        hdul[-1].header['EXTNAME'] = hdul[-1].header['Model_ID']
        hdul[-1].header['Z'] = OH
        hdul[-1].header['qH'] = qH
        hdul[-1].header['LogLsun'] = loglum
        hdul[-1].header['Teff'] = Teff
        hdul[-1].header['nH'] = dens
        hdul[-1].header['Rin'] = r_in
        hdul[-1].header['Rout'] = r_out
        hdul[-1].header['Nzones'] = len(radius)
        hdul[-1].header['Nlines'] = len(M.emis_labels)
        hdul[-1].header['Distance'] = M.distance
        hdul[-1].header['Flux_Ha'] = M.get_emis_vol('H__1_656281A', at_earth=True)
        hdul[-1].header['Source_model'] = M.model_name_s
        nx = len(radius) + 2
        ny = len(M.emis_labels) + 1
        output = np.ndarray((ny, nx), dtype=float)
        out_tab.add_row([hdul[-1].header['Model_ID'], hdul[-1].header['Geometry'], OH, qH, loglum, Teff,
                         dens, r_in, r_out,  len(radius), len(M.emis_labels), M.distance,
                         hdul[-1].header['Flux_Ha'], M.model_name_s])
        for l_id, l in enumerate(M.emis_labels):
            if l[-1] == 'A':
                l_wl = float(l[-7: -1]) / 100.
            elif l[-1] == 'M':
                l_wl = float(l[-7: -1]) / 10.
            else:
                l_wl = np.nan
            output[l_id + 1, 0] = l_wl
            output[l_id + 1, 2:] = M.get_emis(l)
            output[l_id + 1, 1] = M.get_emis_vol(l) / M.get_emis_vol('H__1_656281A')
        output[0, 2:] = radius
        output[0, :2] = np.nan
        hdul[-1].data = output
    hdul.append(fits.BinTableHDU(out_tab, name='Summary'))
    hdul.writeto(fileout, overwrite=True)


def save_continuum_sb99_model(path_to_models, fileout):
    """
    With this script we saved the default Starburst99 output models to the format used in lvmdatasimulator.
    Could be used as an example of how to produce the models in appropriate format
    """
    from scipy.interpolate import interp1d
    models = []
    letters = ['a', 'b', 'c', 'd', 'e']
    for ind in range(5):
        models.append(Table.read(os.path.join(path_to_models,"fig1{}.dat".format(letters[ind])),
                                 header_start=2, format='ascii'))
    wl = models[0]['WAVELENGTH']
    dl = 20.
    l0 = 3000
    l1 = 11000
    z = [0.04, 0.02, 0.008, 0.004, 0.001]
    wlscale = np.linspace(l0, l1, np.round((l1 - l0) / dl).astype(int) + 1)
    models_interp = []
    tab_description = Table(names=['ID', 'Name', 'Description'], dtype=[int, str, str])
    model_index = 0
    for ind in range(5):
        cols = models[ind].colnames
        for col in cols[1:]:
            p = interp1d(wl, 10 ** models[ind][col], assume_sorted=True)
            models_interp.append(p(wlscale))
            tab_description.add_row([model_index, 'Z{:.3f}_t{}'.format(z[ind], col.replace("Myr", "")),
                                     'Starburst99: Z={:.3f}, age={}, M=1e6, inst, '
                                     'IMF_alpha=2.35, Mup=100, Mlow=1'.format(
                                         z[ind], col)])
            model_index += 1

    models_interp = np.array(models_interp)
    hdu = fits.HDUList([fits.PrimaryHDU(models_interp), fits.BinTableHDU(tab_description, name='Summary')])
    hdu[0].header['CRVAL1'] = (l0, "Start wavelength")
    hdu[0].header['CDELT1'] = (dl, "Wavelength step")
    hdu[0].header['CRPIX1'] = 1
    hdu[0].header['CTYPE1'] = "WAV-AWAV"
    hdu[0].header['CUNIT1'] = 'Angstrom'
    hdu.writeto(fileout, overwrite=True)


def set_geocoronal_ha(wave, flux, ha):

    ha_flux = ha * r_to_erg_ha

    log.info(f'Setting Geocoronal Ha to {ha} R ({ha_flux:0.3e})')

    low_cont = np.all([wave > 6560, wave< 6562], axis=0)
    high_cont = np.all([wave > 6564, wave< 6566], axis=0)

    mask_cont = low_cont + high_cont

    # measure baseline continuum
    value_cont = flux[mask_cont].mean()

    mask_line = np.all([wave > 6562, wave< 6564], axis=0)  # location of the line

    # removing the line from the spectrum
    flux[mask_line] = value_cont

    sigma = 0.1  # from a fit of the Ha in the 'LVM_LVM160_SKY_0.dat' file

    I0 = ha_flux.value / (sigma * np.sqrt(2*np.pi))  # from flux to peak

    line = I0 * np.exp(-0.5* (wave[mask_line]-6562.79)**2/sigma**2)

    flux[mask_line] += line

    return flux


def open_sky_file(filename=None, days_moon=None, telescope_name='LVM160',
                  ha=None, area=None):

    if filename is None:
        log.info(f'Simulating the sky emission {days_moon} days from new moon.')
        sky_file = os.path.join(lvmdatasimulator.DATA_DIR, 'sky',
                                    f'LVM_{telescope_name}_SKY_{days_moon}.dat')
    else:
        sky_file = filename
    log.info(f'Using sky file: {sky_file}')

    data = ascii.read(sky_file)
    wave = data["col1"]
    brightness = data["col2"]

    if ha is not None:
        brightness = set_geocoronal_ha(wave, brightness, ha)

    flux = brightness * area  # converting to Fluxes from SBrightness
    return flux, wave


def flam2epp(lam, flam, ddisp):
    """
    Convert flux density [erg/s/cm2/A] to photons per pixel [photons/s/cm2/pixel]

    Args:
        lam (array-like):
            wavelength array associated to the spectrum
        flam (array-like):
            spectrum in units of erg/s/cm2/A
        ddisp (float):
            the pixel scale in A/pixel

    Returns:
        array-like:
            spectrum converted to photons/s/cm2/pixel
    """

    return flam * lam * ddisp / (Constants.h * Constants.c)


def epp2flam(lam, fe, ddisp):
    """
    Convert photons per pixel [photons/s/cm2/pixel] to flux density [erg/s/cm2/A]

    Args:
        lam (array):
            wavelenght axis
        fe (array):
            spectrum in photons/s/cm2/pixel
        ddisp (float):
            dispersion in A/pix


    Returns:
        array:
            spectrum in erg/s/cm2/A
    """

    return fe * Constants.h * Constants.c / (lam * ddisp)


def resample_spectrum(new_wave, old_wave, flux, fast=True):
    """
    Resample spectrum to a new wavelength grid using the spectres package.

    Args:
        new_wave (array-like):
            new wavelength axis.
        old_wave (array-like):
            original wavelength axis
        flux (array-like):
            original spectrum

    Returns:
        array-like:
            spectrum resampled onto the new_wave axis
    """
    if fast:
        f = interp1d(old_wave, flux, fill_value='extrapolate')
        resampled = f(new_wave)
    else:
        resampled = spectres(new_wave, old_wave, flux)

    return resampled


def convolve_for_gaussian(spectrum, fwhm, boundary):
    """
    Convolve a spectrum for a Gaussian kernel.

    Args:
        spectrum (array):
            spectrum to be convolved.
        fwhm (float):
            FWHM of the gaussian kernel.
        boundary (str):
            flag indicating how to handle boundaries.

    Returns:
        array:
            convolved spectrum
    """

    stddev = fwhm / 2.355  # from fwhm to sigma
    size = round_up_to_odd(stddev)  # size of the kernel

    kernel = Gaussian1DKernel(stddev=stddev.value, x_size=size.value)  # gaussian kernel
    return convolve(spectrum, kernel, boundary=boundary)


def yaml_to_plugmap(yaml_file):
    """
    Convert the machine readable fiber file in the format required by the simulator.
    The output is a plugmap.dat file that i saved directly into data/instrument/fibers

    Args:
        yaml_file (string):
            complete name (including path) to the machine readable yaml file with the fiber info

    """

    # creating a big table with the current arrays
    name_science = '/home/econgiu/Data/LVM/lvmdatasimulator/data/instrument/science_array.dat'

    table_old = ascii.read(name_science)

    # open the new machine readable file and rearranging it as astropy table
    with open(yaml_file) as ff:
        fibers = yaml.load(ff)

    table_new = Table(rows=fibers['fibers'], names=['fiberid', 'spectrographid', 'blockid',
                                                    'finblock', 'targettype', 'ifulabel',
                                                    'finifu', 'telescope', 'xpmm', 'ypmxx',
                                                    'ringnum','orig_ifulabel', 'orig_slitlabel',
                                                    'finsector', 'fmap', 'ypix', 'fibstatus'])

    # convert from mm to arcsec -> yaml offsets are in mm
    conv = 37/0.33

    table_new['xpmm'] *= conv
    table_new['ypmxx'] *= conv

    # rotating the science array to match the actual orientation
    angle_rad = 90 * np.pi / 180  # to radians

    # Angle grows moving from north to east!

    newx = table_old['x'] * np.cos(angle_rad) + table_old['y'] * np.sin(angle_rad)
    newy = table_old['y'] * np.cos(angle_rad) - table_old['x'] * np.sin(angle_rad)

    table_old['x'] = newx
    table_old['y'] = newy


    # reordering the info that need to go into the plugmap
    table_new['ring_id'] = np.zeros(len(table_new))
    table_new['fiber_id'] = np.zeros(len(table_new))
    stype = []
    slit = []
    table_new['id'] = np.zeros(len(table_new))

    for i, row in enumerate(table_new):
        # matching the fibers in both table
        dist = np.sqrt((table_old['x'] - table_new['xpmm'][i])**2 +
                       (table_old['y'] - table_new['ypmxx'][i])**2)
        id = np.argmin(dist)

        # associating the correct info to each column
        table_new['ring_id'][i] = table_old['ring_id'][id]
        table_new['fiber_id'][i] = table_old['fiber_id'][id]
        if row['targettype'] == 'science':
            stype.append('science')
        elif row['targettype'] == 'standard':
            stype.append('std')
        elif row['targettype'] == 'SKY' and row['ifulabel'].startswith('A'):
            stype.append('sky1')
        else:
            stype.append('sky2')

        slit.append(f'slit{row["spectrographid"]}')

    table_new['slit'] = slit
    table_new['type'] = stype

    idx = np.arange(648, dtype=int)
    mask = table_new['slit'] == 'slit1'
    table_new['id'][mask] = idx

    mask = table_new['slit'] == 'slit2'
    table_new['id'][mask] = idx

    mask = table_new['slit'] == 'slit3'
    table_new['id'][mask] = idx

    # save the final plugmap.dap
    outtable = table_new['type','ring_id','fiber_id','slit','id'].copy()
    for col in ['ring_id','fiber_id','id']:
        outtable[col] = outtable[col].astype(int)


    # adding the y-position information
    for channel in ['blue', 'red', 'ir']:
        for camera in range(3):

            print(channel, camera+1)
            new_y = compute_y_position(channel, camera+1)

            outtable[f'y_{channel}'] = np.zeros(len(outtable))

            mask = outtable['slit'] == f'slit{camera+1}'
            outtable[f'y_{channel}'][mask] = new_y

    outname = os.path.join(DATA_DIR, 'instrument/fibers/plugmap.dat')
    outtable.write(outname, format='csv', overwrite=True)


def compute_y_position(channel, camera):

    suffix = {'blue': 'b',
              'red': 'r',
              'ir': 'z'}

    trc_name = f'lvm-mtrace-{suffix[channel]}{camera}.fits'
    print(trc_name)
    path = os.path.join(DATA_DIR, 'instrument')

    with fits.open(f'{path}/{trc_name}') as hdu:
            trc_data = hdu[0].data

    '''
    Here I am doing a trick to put the correct spacing in the final image
    I am first assuming that each gap takes the space of 2 fibers.
    I give each fiber an ID which consider these two additional fibers per bundle, then
    I do the interpolation and recover the position of all the fibers (real+manually added).
    At this point, I am removing the non existing values

    '''

    new_y = trc_data[:, 2040]

    return new_y