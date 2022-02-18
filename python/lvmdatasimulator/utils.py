# encoding: utf-8
#
# @Author: Oleg Egorov, Enrico Congiu
# @Date: Nov 12, 2021
# @Filename: field.py
# @License: BSD 3-Clause
# @Copyright: Oleg Egorov, Enrico Congiu

import numpy as np
from astropy.units import UnitConversionError


def round_up_to_odd(f):
    try:
        return np.ceil(f) // 2 * 2 + 1
    except UnitConversionError:
        unit = f.unit
        return (np.ceil(f.value) // 2 * 2 + 1) * unit
