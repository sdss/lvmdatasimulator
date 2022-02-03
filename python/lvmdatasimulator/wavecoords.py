# encoding: utf-8
#
# @Author: Oleg Egorov, Enrico Congiu
# @Date: Nov 12, 2021
# @Filename: field.py
# @License: BSD 3-Clause
# @Copyright: Oleg Egorov, Enrico Congiu

import numpy as np
import astropy.units as u

from dataclasses import dataclass
from abc import ABC, abstractmethod
from functools import cache, cached_property
from astropy.convolution import Gaussian1DKernel

from lvmdatasimulator import ROOT_DIR

@dataclass
class WaveCoord(ABC):
    """
    Abstract class describing the main properties of a WaveCoord object

    """

    @abstractmethod
    def wave(self):
        """Build the wavelength axis"""
        pass

    @abstractmethod
    def start(self):
        """ get the starting wavelength """
        pass

    @abstractmethod
    def end(self):
        """ Get the end wavelength"""
        pass

    @abstractmethod
    def step(self):
        """ Get the step """
        pass


@dataclass
class LinearWave(WaveCoord):
    """ Linear wavelength axis, most for preliminary tests purposes """

    @cached_property
    def wave(self):

        filename = f'{ROOT_DIR}/data/instrument/linear_wave.dat'
        data = np.genfromtxt(filename, skip_header=1, unpack=True)

        return np.arange(data[0], data[1] + data[2], data[2])

    @cached_property
    def start(self):
        return self.wave[0]

    @cached_property
    def end(self):
        return self.wave[-1]

    @cached_property
    def step(self):
        delta = self.wave[1: -1] - self.wave[0: -2]
        return delta.mean()


@dataclass
class BlueWave(WaveCoord):
    """ Wavelength axis of the blue spectrograph """

    @cached_property
    def wave(self):
        pass

    @cached_property
    def start(self):
        pass

    @cached_property
    def end(self):
        pass

    @cached_property
    def step(self):
        pass


@dataclass
class RedWave(WaveCoord):
    """ wavelength axis of the red spectrograph """

    @cached_property
    def wave(self):
        pass

    @cached_property
    def start(self):
        pass

    @cached_property
    def end(self):
        pass

    @cached_property
    def step(self):
        pass

@dataclass
class IRWave(WaveCoord):
    """ wavelength axis of the IR spectrograph """

    @cached_property
    def wave(self):
        pass

    @cached_property
    def start(self):
        pass

    @cached_property
    def end(self):
        pass

    @cached_property
    def step(self):
        pass

