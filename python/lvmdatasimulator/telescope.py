# encoding: utf-8
#
# @Author: Oleg Egorov, Enrico Congiu
# @Date: Nov 12, 2021
# @Filename: field.py
# @License: BSD 3-Clause
# @Copyright: Oleg Egorov, Enrico Congiu

import functools
import astropy.units as u

from dataclasses import dataclass

@dataclass
class LVM160:
    """
    Class summarizing the properties of the telescope. Pretty empty, but for the ETC we need just
    this
    """
    aperture_area: u.cm**2 = 201 * u.cm**2  # not sure where this cames from



