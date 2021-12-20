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
    """Main container for objects in field of view of LVM.

    This is the main class of the simulator of a sources that contains all the functions and
    reproduces the data as it is on the "fake" sky.

    Parameters:
        name (str):
            Name of the current field
        RA (str or None):
            Right Ascension of the center of the field
        Dec (str or None):
            Declination of the center of the field

    Attributes:
        name (str): Name of the current field.
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

    def generate_starlist(self, gmag_limit=17, shift=False, save=True):
        """
        Generate the a list of stars by querying gaia.

        Args:
            gmag_limit (float, optional):
                Maximum magnitude for a star to be included in the list. Defaults to 17.
            shift (bool, optional):
                shift the spectra according to the radial velocity of the stars. Defaults to False.
            save (bool, optional):
                If True save the list of stars to file. Defaults to True.
        """
        self.starlist = StarsList(ra=self.ra, dec=self.dec, radius=self.radius)
        self.starlist.generate(self.wcs, gmag_limit=gmag_limit, shift=shift)

        if save:
            self.starlist.save_to_fits(outname=f'{self.name}_starlist.fits.gz')

    def plot(self, subplots_kw=None, scatter_kw=None):

        fig, ax = plt.subplots(1, 1, subplot_kw=dict(projection=self.wcs))
        self._plot_stars(ax=ax)

        ax.set_xlabel('RA', fontsize=15)
        ax.set_ylabel('Dec', fontsize=15)

        plt.show()

    def _plot_stars(self, ax):

        ax.scatter(self.starlist.stars_table['ra'], self.starlist.stars_table['dec'],
                   transform=ax.get_transform('world'),
                   marker=(5, 1), c='r')
        pass
