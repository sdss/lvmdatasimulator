# encoding: utf-8
#
# @Author: Oleg Egorov, Enrico Congiu
# @Date: Nov 12, 2021
# @Filename: field.py
# @License: BSD 3-Clause
# @Copyright: Oleg Egorov, Enrico Congiu

from functools import cached_property
import functools
import astropy.units as u
import lvmdatasimulator.wavecoords as w

from dataclasses import dataclass
from astropy.io import ascii
from scipy.interpolate import interp1d


from lvmdatasimulator import ROOT_DIR
from abc import ABC, abstractmethod


@dataclass
class Branch:

    name: str
    wavecoord: w.WaveCoord
    lsf_fwhm: u.AA = 0.5 * u.AA
    gain: u.electron / u.adu = 1.0 * u.electron / u.adu
    ron: u.electron = 5.0 * u.electron / u.pix
    dark: u.electron / u.s = 0.001 * u.electron / u.s / u.pix

    def __post_init__(self):
        if self.name not in ['linear', 'red', 'blue', 'ir']:
            raise ValueError(f'{self.name} is not an acepted branch name.')

    @cached_property
    def efficiency(self):
        """create efficiency of the branch"""

        filename = f'{ROOT_DIR}/data/instrument/LVM_ELAM_{self.name}.dat'
        data = ascii.read(filename, names=['col1', 'col2'])
        lam0 = data['col1']
        elam0 = data['col2']
        f = interp1d(lam0, elam0, fill_value='extrapolate')
        return f(self.wavecoord.wave)


class Spectrograph(ABC):

    @property
    @abstractmethod
    def branches(self):
        pass


class LinearSpectrograph(Spectrograph):

    def __init__(self, bundle_name='central'):
        self.bundle_name = bundle_name

    @functools.cached_property
    def branches(self):
        return [Branch(name='linear', wavecoord=w.LinearWave())]


class LVMSpectrograph(Spectrograph):

    def __init__(self, bundle_name='central'):
        self.bundle_name = bundle_name

    @functools.cached_property
    def branches(self):
        return [Branch(name='blue', wavecoord=w.BlueWave()),
                Branch(name='red', wavecoord=w.RedWave()),
                Branch(name='ir', wavecoord=w.IRWave())]
