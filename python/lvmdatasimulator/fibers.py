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
from astropy.table import Table, vstack

from lvmdatasimulator import ROOT_DIR, log


@dataclass
class Fiber:
    """
    Class containing the information on a single LVM fiber
    """
    id: int = 0
    ring: int = 0
    position: int = 0
    x: u.arcsec = 0 * u.arcsec
    y: u.arcsec = 0 * u.arcsec
    diameter: u.arcsec = 35.3 * u.arcsec
    dispersion: u.pix = 3 * u.pix

    def __post_init__(self):
        self.coords = (self.x, self.y)
        self.nypix = 5 * self.dispersion  # size of 2D spectrum

    def __str__(self):
        string = f'Fiber {self.id} in ring {self.ring} located at {self.x}, {self.y}'
        return string


class FiberBundle:
    """
    Class that creates the fiber array used to observe the targeted field of view.

    Args:
    bundle_name (str, optional):
        name of a specific fiber bundle configuration. Available options: 'central', 'full'.
        Defaults to 'central'
    nrings (int, optional):
         _description_. Defaults to None.

    """

    def __init__(self, bundle_name='central', nrings=None, custom_fibers=None, angle=None):

        if bundle_name not in [None, 'central', 'full', 'horizontal', 'diagonals']:
            log.error(f'{bundle_name} is not a valid option for bundle_name')
            raise ValueError()

        self.bundle_name = bundle_name
        self.nrings = nrings
        self.custom_fibers = custom_fibers

        if custom_fibers is not None:
            log.warning('custom_fibers is defined. bundle_name will be ignored.')
        elif bundle_name is None:
            log.error('No bundle definition method is defined.')

        if nrings is not None:
            log.warning('nrings is defined. It will limit the number of fibers selected.')

        if nrings > 24:
            log.warning('The maximum number of rings that can be simulated is 25.')
            self.nrings = 24

        self.angle = angle

        self.fibers = self.build_bundle()

        self.nfibers = len(self.fibers)

    def build_bundle(self):

        fiber_table = self._read_fiber_file()
        if self.custom_fibers is not None:
            log.info('Using custom list of fibers.')
            selected = [row for row in fiber_table
                        if (row['ring_id'], row['fiber_id'])
                        in self.custom_fibers]
            selected = vstack(selected)

        elif self.bundle_name == 'central':

            # open the central fiber
            mask = fiber_table['ring_id'] == 1
            selected = fiber_table[mask].copy()
            log.info('Using only the central fiber.')

        elif self.bundle_name == 'full':
            selected = fiber_table.copy()
            log.info('Using the full hexagon pattern')

        elif self.bundle_name == 'horizontal':
            mask = np.abs(fiber_table['y']) < 1
            selected = fiber_table[mask].copy()
            log.info('Using an horizontal line of fibers.')

        elif self.bundle_name == 'diagonals':
            # hexagon diagonals are lines with slope +- sqrt(3)
            mask1 = np.abs(fiber_table['y'] - np.sqrt(3) * fiber_table['x']) < 1
            mask2 = np.abs(fiber_table['y'] + np.sqrt(3) * fiber_table['x']) < 1
            mask3 = np.abs(fiber_table['y']) < 1
            mask = np.any([mask1, mask2, mask3], axis=0)
            selected = fiber_table[mask].copy()
            log.info('Using a diagonal pattern of fibers.')

        if self.nrings is not None:

            # open the full array until a certain exagonal ring
            log.info(f'Using the inner {self.nrings} exagonal rings.')
            mask = selected['ring_id'] <= self.nrings
            selected = selected[mask].copy()

        # if self.angle is not None:
        #     log.info(f'Rotating the bundle to PA = {self.angle} deg.')
        #     selected = self._rotates(selected)

        fibers = []

        for i, row in enumerate(selected):
            fibers.append(Fiber(i,
                                row['ring_id'],
                                row['fiber_id'],
                                row['x'] * u.arcsec,
                                row['y'] * u.arcsec,
                                row['d'] * u.arcsec,
                                row['disp'] * u.pix))

        return fibers

    @staticmethod
    def _read_fiber_file():
        """
        Read the file containg information on the fibers corresponding to one slit
        TBW
        """

        filename = f'{ROOT_DIR}/data/instrument/full_array.dat'
        table = Table.read(filename, format='ascii.csv')

        return table

    def _rotates(self, table):

        table = table.copy()

        angle_rad = self.angle * np.pi / 180  # to radians

        newx = table['x'] * np.cos(angle_rad) - table['y'] * np.sin(angle_rad)
        newy = table['x'] * np.sin(angle_rad) + table['y'] * np.cos(angle_rad)

        table['x'] = newx
        table['y'] = newy

        return table
