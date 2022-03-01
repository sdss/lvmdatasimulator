# encoding: utf-8
#
# @Author: Oleg Egorov, Enrico Congiu
# @Date: Nov 12, 2021
# @Filename: field.py
# @License: BSD 3-Clause
# @Copyright: Oleg Egorov, Enrico Congiu

import astropy.units as u

from dataclasses import dataclass
from astropy.table import Table

from lvmdatasimulator import ROOT_DIR, log


@dataclass
class Fiber:
    """
    Class containing the information on a single LVM fiber
    """

    id: str = 'central'
    x: u.arcsec = 0 * u.arcsec
    y: u.arcsec = 0 * u.arcsec
    diameter: u.arcsec = 35.3 * u.arcsec
    dispersion: u.pix = 3 * u.pix

    def __post_init__(self):
        self.coords = (self.x, self.y)
        self.nypix = 5 * self.dispersion  # size of 2D spectrum

    def __str__(self):
        string = f'Fiber {self.id} located at {self.x}, {self.y}'
        return string


class FiberBundle:

    def __init__(self, bundle_name='central'):

        self.bundle_name = bundle_name

        self.fibers = self.build_bundle()

        self.nfibers = len(self.fibers)

    def build_bundle(self):

        if self.bundle_name == 'central':
            fiber = Fiber()
            log.info('Creating only the central fiber')
            return [fiber]
        else:
            log.info(f'Creating {self.bundle_name} bundle.')

            return self._read_fiber_file(self.bundle_name)

    @staticmethod
    def _read_fiber_file(slit):
        """
        Read the file containg information on the fibers corresponding to one slit
        TBW
        """

        filename = f'{ROOT_DIR}/data/instrument/{slit}.dat'
        table = Table.read(filename, format='ascii.csv')

        fibers = []

        for row in table:
            fibers.append(Fiber(row['name'],
                                row['x'] * u.arcsec,
                                row['y'] * u.arcsec,
                                row['d'] * u.arcsec,
                                row['disp'] * u.pix))

        return fibers
