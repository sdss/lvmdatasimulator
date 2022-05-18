# encoding: utf-8
#
# @Author: Oleg Egorov, Enrico Congiu
# @Date: Nov 12, 2021
# @Filename: field.py
# @License: BSD 3-Clause
# @Copyright: Oleg Egorov, Enrico Congiu

from dataclasses import dataclass
import numpy as np

from astropy.units import UnitConversionError
from astropy.convolution import convolve_fft
from shapely.geometry import Point, Polygon, box
from pyneb import RedCorr
from lvmdatasimulator import log
from sympy import divisors
from scipy.interpolate import interp2d, RectBivariateSpline
import sys


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

    def set_data(self, newdata):
        self.data = newdata

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
    if unit is not None and not isinstance(mydict[key_to_check], str):
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
            nchunks = 12
        elif max(orig_shape_arcsec) > 1200:
            nchunks = 9
        elif max(orig_shape_arcsec) > 660:
            nchunks = 4
        else:
            nchunks = 1

    if nchunks is 1:
        log.info('Convolving the whole array at once')
    else:
        log.info(f'Dividing the array in {nchunks} chunks')

    # defining the overlap as the size of the kernel + some room
    overlap = 1.1 * np.max(kernel.shape)

    if nchunks > 1:
        log.info(f'Dividing the array in {nchunks} with an overlap of {overlap*pix_size} arcsec')
        # dividing the cube in chuncks before convolving
        chunks = chunksize(to_convolve, nchunks=nchunks, overlap=overlap)
        for chunk in chunks:
            tmp = convolve_fft(chunk.data, kernel, allow_huge=allow_huge,
                               normalize_kernel=normalize_kernel)
            chunk.set_data(tmp)

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
                corners = ((cor_min_dim[i], cor_min_dim[i+1]+1), (cor_max_dim[j], cor_max_dim[j+1]+1))
            else:
                corners = ((cor_max_dim[j], cor_max_dim[j+1]+1), (cor_min_dim[i], cor_min_dim[i+1]+1))
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

    new_cube = np.zeros(orig_shape)

    # loop through the chunks
    for chunk in chunks:
        corner = chunk.original_position
        data = chunk.data
        shape = chunk.shape

        y0 = 0
        y1 = shape[1]
        x0 = 0
        x1 = shape[2]

        if corner[0][0] != 0: y0 += chunk.overlap
        if corner[1][0] != 0: x0 += chunk.overlap
        if corner[0][1] < orig_shape[1]: y1 -= chunk.overlap
        if corner[1][1] < orig_shape[2]: x1 -= chunk.overlap

        new_cube[:, corner[0][0]: corner[0][1], corner[1][0]: corner[1][1]] = \
            data[:, y0: y1, x0: x1]

    return new_cube















