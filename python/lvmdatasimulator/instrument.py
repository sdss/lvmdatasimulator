# encoding: utf-8
#
# @Author: Oleg Egorov, Enrico Congiu
# @Date: Nov 12, 2021
# @Filename: field.py
# @License: BSD 3-Clause
# @Copyright: Oleg Egorov, Enrico Congiu

import astropy.units as u

from dataclasses import dataclass
from astropy.convolution import Gaussian1DKernel


@dataclass
class Instrument:
    """
    Main properties of the LVM instrument. This is just a draft, and contains only what can be
    used to run Kathryn's ETC
    """

    lmin: u.A = 3650 * u.A
    lmax: u.A = 9800 * u.A
    dspax: u.arcsec = 37 * u.arcsec
    dfib: u.px = 3.0 * u.px
    ddisp: u.arcsec / u.pix = 0.5 * u.arcsec / u.pix
    dlsf: u.arcsec = 0.5 * u.arcsec
    gain: u.e / u.adu = 1.0 * u.e / u.adu
    rdn: u.e = 5.0 * u.e
    dark: u.e / u.s = 0.001 * u.e / u.s

    # def line_spread_function(self):
    #     return Gaussian1DKernel(stddev=self.dlsf/2.355/ddisp0,
    #                             x_size=round_up_to_odd(10*self.dlsf/2.355/ddisp0))
