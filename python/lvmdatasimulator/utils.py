# encoding: utf-8
#
# @Author: Oleg Egorov, Enrico Congiu
# @Date: Nov 12, 2021
# @Filename: field.py
# @License: BSD 3-Clause
# @Copyright: Oleg Egorov, Enrico Congiu

import numpy as np
from astropy.units import UnitConversionError
from typing import Union, Tuple
from shapely.geometry import Point, Polygon, box
from shapely.affinity import scale, rotate
from pyneb import RedCorr


def round_up_to_odd(f):
    try:
        return np.ceil(f) // 2 * 2 + 1
    except UnitConversionError:
        unit = f.unit
        return (np.ceil(f.value) // 2 * 2 + 1) * unit


def set_default_dict_values(mydict, key_to_check, default_value):
    """
    Checks the dictionary for the keyword and set the default values if it is missing
    :param mydict: dictionary to be checked
    :param key_to_check: keyword to check
    :param default_value: default value to be used if keyword is missing
    :return: modified dictionary
    """
    if key_to_check not in mydict:
        mydict[key_to_check] = default_value


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
