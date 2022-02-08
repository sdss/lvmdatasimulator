# encoding: utf-8
#
# @Author: Oleg Egorov, Enrico Congiu
# @Date: Nov 12, 2021
# @Filename: field.py
# @License: BSD 3-Clause
# @Copyright: Oleg Egorov, Enrico Congiu

import astropy.units as u
import lvmdatasimulator.wavecoords as w

from dataclasses import dataclass

from lvmdatasimulator import fibers
from abc import ABC, abstractmethod


@dataclass
class Branch:

    name: str
    wavecoord: w.WaveCoord
    lsf_fwhm: u.A = 0.5 * u.A
    gain: u.electron / u.adu = 1.0 * u.electron / u.adu
    ron: u.electron = 5.0 * u.electron
    dark: u.electron / u.s = 0.001 * u.electron / u.s


class Spectrograph(ABC):

    @property
    @abstractmethod
    def brances(self):
        pass

    @property
    @abstractmethod
    def bundle(self):
        pass


class LinearSpectrograph(Spectrograph):

    def __init__(self, bundle='central'):

        self.branches = [Branch(name='linear', wavecoord=w.LinearWave)]

        self.bundle = fibers.FiberBundle(bundle)


class LVMSpectrograph(Spectrograph):

    def __init__(self, bundle='central'):

        self.branches = [Branch(name='blue', wavecoord=w.BlueWave),
                         Branch(name='red', wavecoord=w.RedWave),
                         Branch(name='ir', wavecoord=w.IRWave)]

        self.bundle = fibers.FiberBundle(bundle)
