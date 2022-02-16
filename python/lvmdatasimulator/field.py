# encoding: utf-8
#
# @Author: Oleg Egorov, Enrico Congiu
# @Date: Nov 12, 2021
# @Filename: field.py
# @License: BSD 3-Clause
# @Copyright: Oleg Egorov, Enrico Congiu
# import logging

import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.visualization import ImageNormalize, PercentileInterval, AsinhStretch
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from astropy.visualization.wcsaxes import SphericalCircle
from astropy.wcs import WCS
from lvmdatasimulator.stars import StarsList
from lvmdatasimulator.ism import ISM
import os
from lvmdatasimulator import WORK_DIR, log
from scipy.interpolate import interp1d
import cProfile
from pstats import Stats


fluxunit = u.erg / (u.cm ** 2 * u.s * u.arcsec ** 2)
velunit = u.km / u.s


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
        self.npixels = np.round((self.size.to(u.arcsec) / self.spaxel.to(u.arcsec)).value).astype(int)

        self.wcs = self._create_wcs()
        self.ism = self._create_ism()
        self.ism_map = None
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

    def show(self, subplots_kw=None, scatter_kw=None, apertures=None):
        """
        Display the LVM field with overlaid apertures (if needed). This is a work in progress.

        Args:
            subplots_kw (dict, optional):
                keyword arguments to be passed to plt.subplots. Defaults to None.
            scatter_kw (dict, optional):
                keyword arguments to be passed to plt.scatter. Defaults to None.
            apertures (dict or list of dicts, optional):
                structure defining the position and size of aperture(s) for spectra extraction, values in astropy units
                ({'RA', 'DEC', 'Radius', 'Width', 'Height', 'PA'};
                'Width', 'Height' and 'PA' are ignored if 'Radius' is set)
        """

        fig, ax = plt.subplots(1, 1, subplot_kw=dict(projection=self.wcs))
        if self.ism_map is None:
            self._get_ism_map()
        norm = ImageNormalize(self.ism_map, interval=PercentileInterval(94), stretch=AsinhStretch())
        img = ax.imshow(self.ism_map, norm=norm, cmap=plt.cm.Oranges)
        if self.starlist is not None:
            self._plot_stars(ax=ax)
        cb = plt.colorbar(img)
        cb.set_label(r"F(H$\alpha$), erg s$^{-1}$ cm$^{-2}$", fontsize=15)
        ax.set_xlabel('RA (J2000)', fontsize=15)
        ax.set_ylabel('Dec (J2000)', fontsize=15)

        if type(apertures) in [dict, list, tuple]:
            if type(apertures) == dict:
                apertures = [apertures]
            for ap in apertures:
                if 'RA' not in ap or 'DEC' not in ap or ('Radius' not in ap and
                                                         ('Width' not in ap or 'Height' not in ap)):
                    continue
                if 'Radius' in ap:
                    p = SphericalCircle((ap['RA'], ap['DEC']), ap['Radius'],
                                        edgecolor='green', facecolor='none',
                                        transform=ax.get_transform('fk5'))
                else:
                    if 'PA' not in ap:
                        pa = 0*u.degree
                    else:
                        pa = ap['PA']
                    # !!! TO BE CORRECTED ('RA' and 'DEC' should correspond to the center, and correct for declination)
                    p = Rectangle((ap['RA'].to(u.degree).value, ap['DEC'].to(u.degree).value),
                                  ap['Width'].to(u.degree).value, ap['Height'].to(u.degree).value,
                                  pa.to(u.degree).value,
                                  edgecolor='green', facecolor='none',
                                  transform=ax.get_transform('fk5'))
                ax.add_patch(p)
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

    def _create_ism(self, distance=50 * u.kpc, spec_resolution=0.06 * u.Angstrom,
                    sys_velocity=0*velunit, turbulent_sigma=10.*velunit):
        """
        Create ISM object related to this LVMField
        """
        return ISM(self.wcs, width=self.npixels, height=self.npixels,
                       distance=distance, spec_resolution=spec_resolution, sys_velocity=sys_velocity)

    def add_nebulae(self, list_of_nebulae=None, load_from_file=None):
        """
        Add nebulae to the LVMField.

        Args:
            list_of_nebulae: list containing the dictionaries describing the nebulae:
                example:
                list_of_nebulae = [{type: "Bubble, Filament, DIG, ....",
                                sys_velocity: 0 * u.km/u.s,
                                expansion_velocity: 30 * u.km/u.s,
                                turbulent_sigma: 10 * u.km/u.s,
                                radius: 5 * u.pc,
                                max_brightness: 1e-13 * u.erg / u.cm**2 / u.s,
                                RA: "08h12m13s",
                                DEC: "-20d14m13s",
                                'perturb_degree': 8, # max. order of spherical harmonics to generate inhomogeneities
                                'perturb_amplitude': 0.1, # relative max. amplitude of inhomogeneities,
                                'perturb_scale': 200 * u.pc,  # spatial scale to generate inhomogeneities (DIG only)
                                'cloudy_id': None,  # id of pre-generated Cloudy model
                                'cloudy_params': {'Z': 0.5, 'qH': 49., 'nH': 10, 'Teff': 30000, 'Rin': 0},  #
                                    parameters defining the pre-generated Cloudy model (used if spectrum_id is None)
                                'linerat_constant': False #  if True -> lines ratios doesn't vary across Cloud/Bubble
                                }]
            load_from_file: path (asbolute or relative to work_dir) to the file with previously calculated ISM part
                (preferred if both arguments present)
        """
        loaded = False
        if load_from_file is not None:
            cfile = load_from_file
            if not os.path.isfile(cfile):
                cfile = os.path.join(WORK_DIR, load_from_file)
                if not os.path.isfile(cfile):
                    cfile = None
        else:
            cfile = None
        if cfile is not None:
            loaded = self.ism.load_nebulae(cfile)
            if loaded:
                log.info("Nebulae successfully loaded from file")
                if self.ism_map is not None:
                    self._get_ism_map()
                return True

        if list_of_nebulae is not None:
            loaded = self.ism.generate(list_of_nebulae)
            if loaded and self.ism_map is not None:
                self._get_ism_map()
        if not loaded:
            log.warning("Cannot load the nebulae! Check input parameters.")
            return None

    def _get_ism_map(self):
        """
        Create map of ISM part in Halpha line
        """
        if self.ism.content[0].header['Nobj'] > 0:
            self.ism_map = self.ism.get_map()
        else:
            self.ism_map = np.zeros(shape=(self.npixels, self.npixels), dtype=float)

    def extract_spectra(self, apertures, wl_grid):
        """
        Perform spectra extraction within the given aperture.

        Args:
            apertures (list or tuple of dict, TO BE CHANGED TO APERTURE BUNDLE):
                structure defining the position and size of aperture for spectra extraction, values in astropy units
                ({'RA', 'DEC', 'Radius', 'Width', 'Height', 'PA'};
                'Width', 'Height' and 'PA' are ignored if 'Radius' is set)
            wl_grid (numpy.array): wavelength grid for the resulting spectrum
        """
        if type(apertures) not in [list, tuple]:
            apertures = [apertures]
        spectrum = np.zeros(shape=(len(apertures), len(wl_grid))) * fluxunit
        dl = (wl_grid[1] - wl_grid[0]).to(u.AA)
        xx, yy = np.meshgrid(np.arange(self.npixels), np.arange(self.npixels))
        aperture_mask = np.zeros(shape=(self.npixels, self.npixels), dtype=int)
        for ap_index, aperture in enumerate(apertures):
            if 'RA' not in aperture or 'DEC' not in aperture or ('Radius' not in aperture and
                                                                 ('Width' not in aperture or 'Height' not in aperture)):
                log.warning('Incorrect parameters for aperture #{0}'.format(ap_index + 1))
                continue

            xc, yc = self.wcs.world_to_pixel(SkyCoord(ra=aperture['RA'], dec=aperture['DEC']))
            if 'Radius' in aperture:
                s = (aperture['Radius'].to(u.degree) / self.spaxel.to(u.degree)).value
            else:
                s = (np.max([aperture['Width'].to(u.degree) / 2.,
                             aperture['Height'].to(u.degree) / 2.]) / self.spaxel.to(u.degree)).value

            if (xc - s) < 0 or ((xc + s) >= self.npixels) or ((yc - s) < 0) or ((yc + s) >= self.npixels):
                log.warning('Aperture #{0} for spectra extraction is outside the LVM field'.format(ap_index + 1))
                continue

            if 'Radius' in aperture:
                rec = ((xx - xc) ** 2 + (yy - yc) ** 2) <= (s ** 2)
            else:
                w = (aperture['Width'] / 2. / self.spaxel).value
                h = (aperture['Height'] / 2. / self.spaxel).value
                rec = (xx >= (xc - w)) & (xx <= (xc + w)) & (yy >= (yc - h)) & (yy <= (yc + h))
                # !!!!! ADD ROTATION !!!!
            aperture_mask[rec] = ap_index + 1

            if self.starlist is not None:
                xc_stars = np.round(self.starlist.stars_table['x']).astype(int)
                yc_stars = np.round(self.starlist.stars_table['y']).astype(int)
                stars_id = np.flatnonzero(aperture_mask[yc_stars, xc_stars] == (ap_index + 1))
                for star_id in stars_id:
                    p = interp1d(self.starlist.wave.to(u.AA).value, self.starlist.spectra[star_id])
                    # !!! APPLY EXTINCTION BY DARK NEBULAE TO STARS !!!
                    spectrum[ap_index, :] += (p(wl_grid.value) * dl.value * fluxunit)
        log.info("Start extracting nebular spectra")
        spectrum_ism = self.ism.get_spectrum(wl_grid.to(u.AA), aperture_mask)
        if spectrum_ism is not None:
            spectrum += spectrum_ism
        return spectrum


def run_test(ra=10., dec=-10., spaxel=1/3600., size=1000/60.):
    my_lvmfield = LVMField(ra=ra, dec=dec, size=size, spaxel=spaxel, unit_ra=u.degree, unit_dec=u.degree,
                           unit_size=u.arcmin, unit_spaxel=u.degree)
    my_lvmfield.add_nebulae(load_from_file="/Users/mors/Science/LVM/testneb.fits")
    my_lvmfield.generate_gaia_stars()

    apertures = [{"RA": 10.017*u.degree, "DEC": -10.058*u.degree, "Radius": 30*u.arcsec},
                 {"RA": 10.007 * u.degree, "DEC": -10.05 * u.degree, "Radius": 30 * u.arcsec},
                 {"RA": 09.995 * u.degree, "DEC": -10.0 * u.degree, "Radius": 30 * u.arcsec}]

    my_lvmfield.show(apertures=apertures)

    dlam = 0.06
    l0 = 3650.
    l1 = 9850.
    npix = np.round((l1 - l0) / dlam).astype(int)
    wl_grid = (np.arange(npix) * dlam + l0) * u.AA

    spec = my_lvmfield.extract_spectra(apertures, wl_grid)
    if spec is not None:
        fig, ax = plt.subplots(figsize=(15, 5))
        ax.plot(wl_grid, spec[0], 'k', lw=0.7)
        ax.set_xlabel(r"Wavelength, $'\AA$", fontsize=14)
        ax.set_ylabel(r"Intensity, erg s$^{-1}$ cm$^{-2}$", fontsize=14)
        ax.set_xlim(6550, 6590)
    plt.show()


if __name__ == '__main__':
    run_test()
    # with cProfile.Profile() as pr:
    #     run_test()
    #     with open('/Users/mors/Science/LVM/profiling_stats.prof', 'w') as stream:
    #         stats = Stats(pr, stream=stream)
    #         stats.strip_dirs()
    #         stats.sort_stats('time')
    #         stats.dump_stats('.prof_stats')  # the name of this file is what you have to give to snakeviz
    #         stats.print_stats()
