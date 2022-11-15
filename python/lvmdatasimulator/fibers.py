# encoding: utf-8
#
# @Author: Oleg Egorov, Enrico Congiu
# @Date: Nov 12, 2021
# @Filename: field.py
# @License: BSD 3-Clause
# @Copyright: Oleg Egorov, Enrico Congiu
import os.path

import numpy as np
import astropy.units as u

from dataclasses import dataclass
from astropy.table import Table, vstack

from lvmdatasimulator import DATA_DIR, log


@dataclass
class Fiber:
    """
    Low level object representing a single LVM fiber

    Attributes:

        id (int, optional):
            Unique ID of the fiber in the current bundle. Defaults to 0.
        ring (int, optional):
            hexagonal ring where the fiber is positioned. Defaults to 0.
        position (int, optional):
            position of the fiber in the hexagonal ring considered. Defaults to 0.
        x (astropy.quantity, optional):
            Offset of the fiber in the x direction with respect to the center of the fiber bundle.
            Defaults to 0 * u.arcsec.
        y (astropy.quantity, optional):
            Offset of the fiber in the y direction with respect to the center of the fiber bundle.
            Defaults to 0 * u.arcsec.
        diameter (astropy.quantity, optional):
            Diameter of the fiber on the sky. Defaults to 35.3 * u.arcsec
        dispersion (astropy.quantity, optional):
            Dispersion caused by the fiber in the spatial direction. Defaults to 3 * u.pix
        type (str, optional):
            type of fiber (std, science, sky). Defaults to science.

    """
    id: int = 0
    ring: int = 0
    position: int = 0
    x: u.arcsec = 0 * u.arcsec
    y: u.arcsec = 0 * u.arcsec
    diameter: u.arcsec = 35.3 * u.arcsec
    dispersion: u.pix = 3 * u.pix
    type: str = 'science'

    def __post_init__(self):
        self.coords = (self.x, self.y)
        self.nypix = 5 * self.dispersion  # size of 2D spectrum

    def __str__(self):
        string = f'Fiber {self.id} in ring {self.ring}, position {self.position}' +\
            f'located at {self.x}, {self.y}'
        return string

    def to_table(self):
        """
        return the main properties of the fiber as an astropy table.

        Returns:
            astropy.table.Table:
                Table containing the information about the fiber.
        """

        out = {
            'id': [self.id],
            'ring': [self.ring],
            'position': [self.position],
            'x': [self.x],
            'y': [self.y],
            'diameter': [self.diameter],
            'dispersion': [self.dispersion]
        }

        return Table(out)


class FiberBundle:
    """
    Class that creates the fiber array used to observe the targeted field of view.

    Args:
    bundle_name (str, optional):
        name of a specific fiber bundle configuration. Available options: 'central', 'full'.
        Defaults to 'central'
    nrings (int, optional):
        last hexagonal ring to be considered when building the fiber bundle. If None,
        the full array will be simulated. Defaults to None.
    custom_fibers (list, optional):
        list of tuples containing an arbitrary configuration of fibers. The fibers are identified
        by their hexagonal ring and by the position in their hexagonal ring
    angle (float, optional):
        rotation to be applied to the final bundle of fiber in degrees. If None, no rotation is
        applied. Defaults to None
    """

    def __init__(self, bundle_name='central', nrings=None, custom_fibers=None, angle=None,
                 max_fibers=1944, max_obj_fibers=1801, max_sky_fibers=119, max_std_fibers=24):

        if bundle_name not in [None, 'central', 'full', 'horizontal', 'diagonals']:
            log.error(f'{bundle_name} is not a valid option for bundle_name')
            raise ValueError()

        self.bundle_name = bundle_name
        self.nrings = nrings
        self.custom_fibers = custom_fibers
        self.max_fibers = max_fibers
        self.max_obj_fibers = max_obj_fibers
        self.max_sky_fibers = max_sky_fibers
        self.max_std_fibers = max_std_fibers

        if custom_fibers is not None:
            log.warning('custom_fibers is defined. bundle_name will be ignored.')
        elif bundle_name is None:
            log.error('No bundle definition method is defined.')

        if nrings is not None:
            log.warning('nrings is defined. It will limit the number of fibers selected.')
            if nrings > 25:
                log.warning('The maximum number of rings that can be simulated is 25.')
                self.nrings = 25
            if nrings < 1:
                log.warning('The ring number goes from 1 to 25. Setting it to 1.')
                self.nrings = 1

        self.angle = angle

        self.build_bundles()

        self.nfibers = len(self.fibers_science)

    def build_bundles(self):
        """
        Read the database containing the informations on the fibers and setup the bundle to be
        used for the observations.

        Returns:
            list:
                list of fibers to be simulated.
        """

        #### create science fiber bundle

        fiber_table = self._read_fiber_file(name='science_array.dat')
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

            # open the full array until a certain hexagonal ring
            log.info(f'Using the inner {self.nrings} hexagonal rings.')
            mask = selected['ring_id'] <= self.nrings
            selected = selected[mask].copy()

        if self.angle is not None:
            log.info(f'Rotating the bundle to PA = {self.angle} deg.')
            selected = self._rotates(selected)

        fibers_science = self._generate_fibers(selected)

        self.fibers_science = fibers_science
        self.fibers_table_science = selected

        ### sky fiber bundles
        fibers_table_sky1 = self._read_fiber_file(name='sky1_array.dat')
        fibers_table_sky2 = self._read_fiber_file(name='sky2_array.dat')

        fibers_sky1 = self._generate_fibers(fibers_table_sky1)
        fibers_sky2 = self._generate_fibers(fibers_table_sky2)

        self.fibers_sky1 = fibers_sky1
        self.fibers_table_sky1 = fibers_table_sky1

        self.fibers_sky2 = fibers_sky2
        self.fibers_table_sky2 = fibers_table_sky2

        ### standard fiber bundle

        fibers_table_std = self._read_fiber_file(name='std_array.dat')

        fibers_std = self._generate_fibers(fibers_table_std)

        self.fibers_std = fibers_std
        self.fibers_table_std = fibers_table_std


    @staticmethod
    def _read_fiber_file(name='science_array.dat'):
        """
        Reads the file containing the information on each fiber and it returns it as an astropy
        table

        Returns:
            astropy.table.Table:
                table containing the informations about each fiber.
        """

        filename = os.path.join(DATA_DIR, 'instrument', name)
        table = Table.read(filename, format='ascii.csv')

        return table

    def _rotates(self, table):
        """
        Apply a rotation to the selected fibers

        Args:
            table (astropy.table.Table):
                Table containing the information on the selected fibers. Only the 'x' and 'y'
                columns are used.

        Returns:
            astropy.table.Table:
                Table containing the information on the fibers updated after the rotation.
        """

        table = table.copy()

        angle_rad = self.angle * np.pi / 180  # to radians

        # Angle grows moving from north to east!
        newx = table['x'] * np.cos(angle_rad) + table['y'] * np.sin(angle_rad)
        newy = table['y'] * np.cos(angle_rad) - table['x'] * np.sin(angle_rad)

        table['x'] = newx
        table['y'] = newy

        return table

    @staticmethod
    def _generate_fibers(table):

        out = []
        for i, row in enumerate(table):
            out.append(Fiber(i,
                                row['ring_id'],
                                row['fiber_id'],
                                row['x'] * u.arcsec,
                                row['y'] * u.arcsec,
                                row['d'] * u.arcsec,
                                row['disp'] * u.pix,
                                row['type']))
        return out
