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
from astropy.io import fits
from astropy.coordinates import SkyCoord
from astropy.visualization import ImageNormalize, PercentileInterval, AsinhStretch
import matplotlib.pyplot as plt
from astropy.visualization.wcsaxes import SphericalCircle
from astropy.wcs import WCS
from lvmdatasimulator.stars import StarsList
from lvmdatasimulator.ism import ISM
import os
from lvmdatasimulator import WORK_DIR, log
from scipy.interpolate import interp1d
# import cProfile
# from pstats import Stats


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
        self.coord = SkyCoord(self.ra, self.dec)
        self.size = size * unit_size
        self.radius = self.size / 2  # to generate the star list
        self.spaxel = spaxel * unit_spaxel
        self.npixels = np.round((self.size.to(u.arcsec) /
                                 self.spaxel.to(u.arcsec)).value).astype(int)

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

    def show(self, subplots_kw=None, scatter_kw=None, fibers=None):
        """
        Display the LVM field with overlaid apertures (if needed). This is a work in progress.

        Args:
            subplots_kw (dict, optional):
                keyword arguments to be passed to plt.subplots. Defaults to None.
            scatter_kw (dict, optional):
                keyword arguments to be passed to plt.scatter. Defaults to None.
            fibers (dict or list of fibers, optional):
                structure defining the position and size of aperture(s) for spectra extraction.
        """

        fig, ax = plt.subplots(1, 1, subplot_kw=dict(projection=self.wcs))
        if self.ism_map is None:
            self._get_ism_map(wavelength=6562.81)
        norm = ImageNormalize(self.ism_map, interval=PercentileInterval(98),
                              stretch=AsinhStretch())
        img = ax.imshow(self.ism_map, norm=norm, cmap=plt.cm.Oranges)
        if self.starlist is not None:
            self._plot_stars(ax=ax)
        cb = plt.colorbar(img)
        cb.set_label(r"F(H$\alpha$), erg s$^{-1}$ cm$^{-2}$", fontsize=15)
        ax.set_xlabel('RA (J2000)', fontsize=15)
        ax.set_ylabel('Dec (J2000)', fontsize=15)

        if fibers is not None:
            for fiber in fibers:
                fiber_coord = self.coord.spherical_offsets_by(fiber.x, fiber.y)
                p = SphericalCircle((fiber_coord.ra, fiber_coord.dec), fiber.diameter / 2,
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
                    sys_velocity=0 * velunit, turbulent_sigma=10. * velunit):
        """
        Create ISM object related to this LVMField
        """
        return ISM(self.wcs,
                   width=self.npixels,
                   height=self.npixels,
                   turbulent_sigma=turbulent_sigma,
                   distance=distance,
                   spec_resolution=spec_resolution,
                   sys_velocity=sys_velocity)

    def get_map(self, wavelength_range=None, save_file=None):
        self.ism_map = np.zeros(shape=(self.npixels, self.npixels), dtype=float)
        if wavelength_range is None:
            log.info("Input map is produced for default lambda = 6562.81")
            wavelength_range = 6562.81
        wavelength_range = np.atleast_1d(wavelength_range)
        if len(wavelength_range) == 1:
            wavelength_range = np.array([wavelength_range[0] - 0.01, wavelength_range[1] + 0.01])
        if self.starlist is not None:
            xc_stars = np.round(self.starlist.stars_table['x']).astype(int)
            yc_stars = np.round(self.starlist.stars_table['y']).astype(int)
            wl_grid = np.linspace(wavelength_range[0], wavelength_range[1], 10)
            for star_id, xy in enumerate(zip(xc_stars, yc_stars)):
                if (xy[0] >= self.npixels) or (xy[1] >= self.npixels) or (xy[0] < 0) or (xy[1] < 0):
                    continue
                p = interp1d(self.starlist.wave.to(u.AA).value, self.starlist.spectra[star_id], bounds_error=False,
                             fill_value='extrapolate')
                # !!! APPLY EXTINCTION BY DARK NEBULAE TO STARS !!!
                self.ism_map[xy[1], xy[0]] += np.sum(p(wl_grid) * (wl_grid[1] - wl_grid[0]))

        if self.ism.content[0].header['Nobj'] > 0:
            self.ism_map += self.ism.get_map(wavelength=wavelength_range, get_continuum=True)

        if save_file is not None:
            header = self.wcs.to_header()
            header['LAMRANGE'] = ("{0}-{1}".format(wavelength_range[0], wavelength_range[1]),
                                  "Wavelength range used for image extraction")
            fits.writeto(save_file, data=self.ism_map, header=header, overwrite=True)
            log.info("Input image in {0}-{1}AA wavelength range "
                     "is saved to {2}".format(wavelength_range[0], wavelength_range[1], save_file))



    def add_nebulae(self, list_of_nebulae=None, load_from_file=None, save_nebulae=None):
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
                                'perturb_degree': 8, # max. order of spherical harmonics
                                                       to generate inhomogeneities
                                'perturb_amplitude': 0.1, # relative max.
                                                            amplitude of inhomogeneities,
                                'perturb_scale': 200 * u.pc,  # spatial scale to generate
                                                                inhomogeneities (DIG only)
                                'cloudy_id': None,  # id of pre-generated Cloudy model
                                'cloudy_params': {'Z': 0.5,
                                                  'qH': 49.,
                                                  'nH': 10,
                                                  'Teff': 30000,
                                                  'Rin': 0},  # parameters defining the
                                        pre-generated Cloudy model (used if spectrum_id is None)
                                'linerat_constant': False #  if True -> lines ratios doesn't
                                                                vary across Cloud/Bubble
                                }]
            load_from_file: path (asbolute or relative to work_dir) to the file with previously
                            calculated ISM part (preferred if both load_from_file and
                            list_of_nebulae are present)
            save_nebulae: path (asbolute or relative to work_dir) where to save fits with
                          calculated ISM (only used if list_of_nebulae is present and
                          load_from_file is not)

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
            if loaded and save_nebulae is not None:
                if not (save_nebulae.startswith('/') or save_nebulae.startswith(r'\\')):
                    save_nebulae = os.path.join(WORK_DIR, save_nebulae)
                self.ism.save_ism(save_nebulae)
        if not loaded:
            log.warning("Cannot load the nebulae! Check input parameters.")
            return None

    def _get_ism_map(self, wavelength=6562.81):
        """
        Create map of ISM part in desired line or wavelength range (default = Halpha)
        """
        if self.ism.content[0].header['Nobj'] > 0:
            self.ism_map = self.ism.get_map(wavelength=wavelength)
        else:
            self.ism_map = np.zeros(shape=(self.npixels, self.npixels), dtype=float)

    def extract_spectra(self, fibers, wl_grid):
        """
        Perform spectra extraction within the given aperture.

        Args:
            fibers (list or tuple fiber objects):
                structure defining the position and size of aperture for spectra extraction
            wl_grid (numpy.array): wavelength grid for the resulting spectrum
        """

        spectrum = np.zeros(shape=(len(fibers), len(wl_grid))) * fluxunit * u.arcsec ** 2
        fiber_id = []
        dl = (wl_grid[1] - wl_grid[0]).to(u.AA)
        xx, yy = np.meshgrid(np.arange(self.npixels), np.arange(self.npixels))
        aperture_mask = np.zeros(shape=(self.npixels, self.npixels), dtype=int)
        fibers_coords = np.zeros(shape=(len(fibers), 3), dtype=float)
        for index, fiber in enumerate(fibers):
            fiber_id.append(fiber.id)
            # I have to check this holds
            cur_fiber_coord = self.coord.spherical_offsets_by(fiber.x, fiber.y)
            xc, yc = self.wcs.world_to_pixel(cur_fiber_coord)

            s = (fiber.diameter.to(u.degree) / self.spaxel.to(u.degree)).value / 2
            fibers_coords[index, :] = [xc, yc, s]
            if (xc - s) < 0 or ((xc + s) >= self.npixels) \
                    or ((yc - s) < 0) or ((yc + s) >= self.npixels):

                log.warning('Aperture #{0} for spectra extraction is outside of the LVM field'
                            .format(index + 1))
                continue

            rec = ((xx - xc) ** 2 + (yy - yc) ** 2) <= (s ** 2)
            aperture_mask[rec] = index + 1

            if self.starlist is not None:
                xc_stars = np.round(self.starlist.stars_table['x']).astype(int)
                yc_stars = np.round(self.starlist.stars_table['y']).astype(int)
                stars_id = np.flatnonzero(aperture_mask[yc_stars, xc_stars] == (index + 1))
                for star_id in stars_id:
                    p = interp1d(self.starlist.wave.to(u.AA).value, self.starlist.spectra[star_id], bounds_error=False,
                                 fill_value='extrapolate')
                    # !!! APPLY EXTINCTION BY DARK NEBULAE TO STARS !!!
                    spectrum[index, :] += (p(wl_grid.value) * dl.value * fluxunit * u.arcsec ** 2)
        if np.max(aperture_mask) < fibers_coords.shape[0]:
            fibers_coords = fibers_coords[:np.max(aperture_mask), :]
        log.info("Start extracting nebular spectra")
        spectrum_ism = self.ism.get_spectrum(wl_grid.to(u.AA), aperture_mask, fibers_coords)
        if spectrum_ism is not None:
            spectrum[: len(spectrum_ism), :] += spectrum_ism
        return np.array(fiber_id), spectrum / dl.value / u.AA

      
def run_test(ra=10., dec=-10., spaxel=1 / 3600., size=1000 / 60.):
    # my_nebulae = [
    #         {"type": 'Bubble', 'expansion_velocity': 45 * u.km/u.s,
    #          'turbulent_sigma': 20 * u.km/u.s,
    #          'radius': 10 * u.pc,
    #          'max_brightness': 3e-16 * u.erg / u.cm**2 / u.s / u.arcsec ** 2,
    #          'RA': '00h39m40s', 'DEC': "-10d02m13s",
    #          'cloudy_params': {'Z': 0.4, 'qH': 50., 'nH': 30, 'Teff': 50000, 'Rin': 0},
    #          },  # 'perturb_amplitude': 0.1, 'perturb_order': 8},
    #         {"type": 'Cloud',
    #          'radius': 12 * u.pc,
    #          'max_extinction': 0.6 * u.mag,
    #          'RA': "00h39m10s", 'DEC': "-10d00m00s", 'zorder': 2},
    #         {"type": 'Filament',
    #          'length': 50 * u.pc, 'width': 1. * u.pc, 'PA': -35 * u.degree,
    #          'max_brightness': 3e-17 * u.erg / u.cm**2 / u.s / u.arcsec ** 2,
    #          'cloudy_params': {'Z': 0.2, 'qH': 48., 'nH': 30, 'Teff': 30000, 'Rin': 0},
    #          'RA': "00h40m10s", 'DEC': "-10d01m20s", 'zorder': 3},
    #         {"type": 'Bubble', 'expansion_velocity': 25 * u.km/u.s,
    #          'turbulent_sigma': 20 * u.km/u.s,
    #          'radius': 25 * u.pc,
    #          'max_brightness': 1e-16 * u.erg / u.cm**2 / u.s / u.arcsec ** 2,
    #          'RA': "00h40m05s", 'DEC': "-10d04m00s",
    #          'cloudy_params': {'Z': 0.6, 'qH': 49., 'nH': 30, 'Teff': 30000, 'Rin': 0},
    #          'perturb_amplitude': 0.4, 'perturb_order': 8},
    #         {"type": 'Bubble', 'expansion_velocity': 55 * u.km / u.s,
    #          'turbulent_sigma': 20 * u.km / u.s,
    #          'radius': 15 * u.pc,
    #          'max_brightness': 2e-16 * u.erg / u.cm ** 2 / u.s / u.arcsec ** 2,
    #          'RA': "00h40m00s", 'DEC': "-09d56m00s",
    #          'cloudy_params': {'Z': 0.4, 'qH': 50., 'nH': 30, 'Teff': 70000, 'Rin': 0},
    #          'perturb_amplitude': 0.1, 'perturb_order': 8},
    #         {"type": 'DIG', 'max_brightness': 1e-17 * u.erg / u.cm ** 2 / u.s / u.arcsec ** 2,
    #          'perturb_amplitude': 0.1, 'perturb_scale': 200 * u.pc}
    #         ]

    my_lvmfield = LVMField(ra=ra, dec=dec, size=size, spaxel=spaxel, unit_ra=u.degree,
                           unit_dec=u.degree, unit_size=u.arcmin, unit_spaxel=u.degree)
    # my_lvmfield.add_nebulae(my_nebulae, save_nebulae="/Users/mors/Science/LVM/testneb_v2.fits")
    my_lvmfield.add_nebulae(load_from_file="/Users/mors/Science/LVM/testneb_v2.fits")
    my_lvmfield.generate_gaia_stars()

    apertures = [{"RA": 10.018 * u.degree, "DEC": -10.059 * u.degree, "Radius": 30 * u.arcsec},
                 # {"RA": 10.007 * u.degree, "DEC": -10.05 * u.degree, "Radius": 30 * u.arcsec},
                 # {"RA": 09.995 * u.degree, "DEC": -10.0 * u.degree, "Radius": 30 * u.arcsec},
                 # {"RA": 09.985 * u.degree, "DEC": -9.95 * u.degree, "Radius": 30 * u.arcsec},
                 ]

    my_lvmfield.show(fibers=apertures)

    dlam = 0.06
    l0 = 3650.
    l1 = 9850.
    npix = np.round((l1 - l0) / dlam).astype(int)
    wl_grid = (np.arange(npix) * dlam + l0) * u.AA

    spec = my_lvmfield.extract_spectra(apertures, wl_grid)
    if spec is not None:
        _ = plt.figure(figsize=(12, 10))
        for i in range(2):
            ax = plt.subplot(211 + i)
            ax.plot(wl_grid, spec[0], 'k', lw=0.7)
            ax.set_xlabel(r"Wavelength, $'\AA$", fontsize=14)
            ax.set_ylabel(r"Intensity, erg s$^{-1}$ cm$^{-2}$", fontsize=14)
            if i == 0:
                ax.set_xlim(6500, 6600)
            else:
                ax.set_xlim(3700, 9600)

    plt.show()


if __name__ == '__main__':
    run_test()
    # with cProfile.Profile() as pr:
    #     run_test()
    #     with open('/Users/mors/Science/LVM/profiling_stats.prof', 'w') as stream:
    #         stats = Stats(pr, stream=stream)
    #         stats.strip_dirs()
    #         stats.sort_stats('time')
    #         stats.dump_stats('.prof_stats')  # the name you have to give to snakeviz
    #         stats.print_stats()

