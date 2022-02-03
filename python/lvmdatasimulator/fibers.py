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
from functools import cached_property
from astropy.convolution import Gaussian1DKernel

from lvmdatasimulator.utils import round_up_to_odd
from lvmdatasimulator import ROOT_DIR, log


@dataclass
class Fiber:
    """
    Class containing the information on a single LVM fiber
    """

    id: str = 'central'
    x: u.arcsec = 0 * u.arcsec
    y: u.arcsec = 0 * u.arcsec
    diameter: u.arcsec = 37 * u.arcsec
    dispersion: u.pix = 3 * u.pix

    def __post_init(self):
        self.coords = (self.x, self.y)

    def __str__(self):
        string = f'Fiber {self.id} located at {self.x}, {self.y}'
        return string


class FiberBundle:

    def __init__(self, bundle_type='central'):

        if bundle_type not in ['central', 'slit1', 'slit2', 'slit3', 'full']:
            error = f'{bundle_type} is not an accepted bundle.'+ \
                    'Allowed values: central, slit1, slit2, slit3, full.'
            log.error(error)
            raise ValueError(error)

        self.bundle_type = bundle_type

        self.fibers = self.build_bundle()

    def build_bundle(self):

        if self.bundle_type == 'central':
            fiber = Fiber()
            log.info('Creating only the central fiber')
            return [fiber]

        elif self.bundle_type == 'full':
            log.info('Creating the full fiber bundle')
            fiber1 = self._read_fiber_file('slit1')
            fiber2 = self._read_fiber_file('slit2')
            fiber3 = self._read_fiber_file('slit3')
            return fiber1 + fiber2 + fiber3

        else:
            log.info(f'Creating only {self.bundle_type}.')

            return self._read_fiber_file(self.bundle_type)

    @staticmethod
    def _read_fiber_file(slit):
        """
        Read the file containg information on the fibers corresponding to one slit
        TBW
        """
        return []








