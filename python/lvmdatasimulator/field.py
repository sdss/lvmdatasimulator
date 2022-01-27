# encoding: utf-8
#
# @Author: Oleg Egorov, Enrico Congiu
# @Date: Nov 12, 2021
# @Filename: field.py
# @License: BSD 3-Clause
# @Copyright: Oleg Egorov, Enrico Congiu

import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt

from astropy.wcs import WCS

from .stars import StarsList


class LVMField:
    """
    Main container for objects in the field of view of LVM.

    This is the main class of the simulator of a sources that contains all the functions and
    reproduces the data as it is on the "fake" sky.

    Parameters:
        ra (float):
            Right ascension of the center of the field. This parameter is expected in degree,
            but the unit can be changed with the unit_ra parameter.
        dec (float):
            Declination of the center of the field. This parameter is expected in degree, but the
            unit can be changed with the unit_dec parameter.
        size (float):
            Size of the field to be created. The filed will be a square with side "size". size is
            expected in arcmin, but the unit can be changed using unit_size.
        spaxel (float):
            Size of the squared spatial resolution element to be used to create the wcs object
            associated to the field. It is expected to be in arcsec, but the unit can be changed
            by using unit_spaxel.
        unit_ra (astropy.unit):
            Physical units of the right ascension. Defaults to u.deg.
        unit_dec (astropy.unit):
            Physical units of the declination. Defaults to u.deg.
        unit_size (astropy.unit):
            Physical units of the size. Defaults to u.arcmin.
        unit_spaxel (astropy.unit):
            Physical units of the spaxel. Defaults to u.arcsec.
        name (str):
            Name of the field. Defaults to 'LVM_field'

    Attributes:
        name (str):
            Name of the current field.
        RA (str or None):
            Right Ascension of the center of the field
        Dec (str or None):
            Declination of the center of the field

    """

    def __init__(self, ra, dec, size, spaxel, unit_ra=u.deg, unit_dec=u.deg,
                 unit_size=u.arcmin, unit_spaxel=u.arcsec,
                 name='LVM_field'):

        self.name = name
        self.ra = ra * unit_ra
        self.dec = dec * unit_dec
        self.size = size * unit_size
        self.radius = self.size / 2  # to generate the star list
        self.spaxel = spaxel * unit_spaxel
        self.npixels = self.size.to(u.arcsec) / self.spaxel.to(u.arcsec)

        self.wcs = self._create_wcs()

        self.starlist = None

    def _create_wcs(self):
        """
        Create a wcs object that can be used to generate the elements of the field.

        The reference point is at the center of the field, the pixel size is defined by the user.

        Returns:
            astropy.wcs:
                wcs object with the desired quantities
        """

        # initializing the wcs object
        wcs = WCS(naxis=2)

        # setting up the different fields
        wcs.wcs.crpix = [self.npixels / 2, self.npixels / 2]
        wcs.wcs.cdelt = np.array([-self.spaxel.to(u.deg).value, self.spaxel.to(u.deg).value])
        wcs.wcs.crval = [self.ra.to(u.deg).value, self.dec.to(u.deg).value]
        wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]

        return wcs

    def generate_gaia_stars(self, gmag_limit=17, shift=False, save=True):
        """
        Generate a list of stars by querying gaia.

        Args:
            gmag_limit (float, optional):
                Maximum magnitude for a star to be included in the list. Defaults to 17.
            shift (bool, optional):
                shift the spectra according to the radial velocity of the stars. Defaults to False.
            save (bool, optional):
                If True save the list of stars to file. Defaults to True.
        """
        self.starlist = StarsList(ra=self.ra, dec=self.dec, radius=self.radius)
        self.starlist.generate_gaia(self.wcs, gmag_limit=gmag_limit, shift=shift)

        if save:
            self.starlist.save_to_fits(outname=f'{self.name}_starlist.fits.gz')

    def generate_single_stars(self, parameters={}, shift=False, save=False):
        """
        Generate a list of stars by manually providing the basic informations

        Args:
            shift (bool, optional):
                apply the shift due to the velocity of the stars if available. Defaults to False.
            save (bool, optional):
                Save the starlist to file. Defaults to False.
            parameters (dictionary, optional):
                Dictionary containig the parameters to pass to the StarsList.add_star() method.
                Since more than one star can be generate simultaneously with this method, each
                type of parameter must be provided as lists. If no dictionary is provided, a
                single star is created with standard parameters.
                Defaults to {}.
        """

        self.starlist = StarsList(ra=self.ra, dec=self.dec, radius=self.radius)
        self.starlist.generate_stars_manually(self.wcs, parameters, shift=shift)

        if save:
            self.starlist.save_to_fits(outname=f'{self.name}_starlist.fits.gz')

    def open_starlist(self, filename, directory='./'):
        """
        Open an existing file list from a fits file.

        Args:
            filename (str):
                Name of the fits file containing the StarList.
            directory (str, optional):
                Directory containing the file. Defaults to './'.
        """

        self.starlist = StarsList(filename=filename, dir=directory)

    def plot(self, subplots_kw=None, scatter_kw=None):
        """
        Plots the LVM field. This is a work in progress.

        Args:
            subplots_kw (dict, optional):
                keyword arguments to be passed to plt.subplots. Defaults to None.
            scatter_kw (dict, optional):
                keyword arguments to be passed to plt.scatter. Defaults to None.
        """

        fig, ax = plt.subplots(1, 1, subplot_kw=dict(projection=self.wcs))
        self._plot_stars(ax=ax)

        ax.set_xlabel('RA', fontsize=15)
        ax.set_ylabel('Dec', fontsize=15)

        plt.show()

    def _plot_stars(self, ax):
        """
        Create a scatter plot showing the position of the stars. This is just a preliminary
        thing.

        Args:
            ax (axis):
                axis where to plot the position of the stars.
        """

        ax.scatter(self.starlist.stars_table['ra'], self.starlist.stars_table['dec'],
                   transform=ax.get_transform('world'),
                   marker=(5, 1), c='r')
        pass
