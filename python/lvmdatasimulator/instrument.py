# encoding: utf-8
#
# @Author: Oleg Egorov, Enrico Congiu
# @Date: Nov 12, 2021
# @Filename: field.py
# @License: BSD 3-Clause
# @Copyright: Oleg Egorov, Enrico Congiu

import numpy as np
import functools
import astropy.units as u
import lvmdatasimulator.wavecoords as w

from dataclasses import dataclass
from astropy.convolution import Gaussian1DKernel

from lvmdatasimulator.utils import round_up_to_odd
from lvmdatasimulator import fibers


@dataclass
class Branch:

    wavecoord: w.WaveCoord
    lsf_fwhm: u.A = 0.5 * u.A
    gain: u.electron / u.adu = 1.0 * u.electron / u.adu
    ron: u.electron = 5.0 * u.electron
    dark: u.electron / u.s = 0.001 * u.electron / u.s


class LinearSpectrograph:

    def __init__(self, bundle='central'):

        self.branches = [Branch(wavecoord=w.LinearWave)]

        self.bundle = fibers.FiberBundle(bundle)


class LVMSpectrograph:

    def __init__(self, bundle='central'):

        self.branches = [Branch(wavecoord=w.BlueWave),
                         Branch(wavecoord=w.RedWave),
                         Branch(wavecoord=w.IRWave)]

        self.bundle = fibers.FiberBundle(bundle)
