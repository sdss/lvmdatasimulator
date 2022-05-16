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

import lvmdatasimulator
from lvmdatasimulator.stars import StarsList
from lvmdatasimulator.ism import ISM
import os
from lvmdatasimulator import log
from lvmdatasimulator.utils import assign_units
from scipy.interpolate import interp1d


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
        ism_params (dict):
            Dictionary with the parameters defining the properties of the ISM
            (distance, spec_resolution, sys_velocity, turbulent_sigma)

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
                 name='LVM_field', ism_params=None):

        self.name = name
        self.ra = ra * unit_ra
        self.dec = dec * unit_dec
        self.coord = SkyCoord(self.ra, self.dec)
        self.size = size * unit_size
        self.radius = self.size / 2  # to generate the star list
        self.spaxel = spaxel * unit_spaxel

        assign_units(self, ['spaxel', 'radius', 'size', 'ra', 'dec'],
                     [u.arcsec, u.arcmin, u.arcmin, u.degree, u.degree])

        self.npixels = np.round((self.size.to(u.arcsec) /
                                 self.spaxel.to(u.arcsec)).value/2.).astype(int) * 2 + 1
        # made this odd for consistency with demands for some Nebulae (e.g. DIG)

        self.wcs = self._create_wcs()
        self.ism_params = {'distance': 50 * u.kpc, 'spec_resolution': 0.06 * u.Angstrom,
                           'sys_velocity': 0 * velunit, 'turbulent_sigma': 10. * velunit}
        if type(ism_params) is dict:
            for k, v in ism_params.items():
                self.ism_params[k] = v
        self.ism = self._create_ism(**self.ism_params)
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

    def generate_gaia_stars(self, gmag_limit=17, shift=False, save=True, filename=None,
                            directory=lvmdatasimulator.WORK_DIR):
        """
        Generate a list of stars by querying gaia.

        Args:
            gmag_limit (float, optional):
                Maximum magnitude for a star to be included in the list. Defaults to 17.
            shift (bool, optional):
                shift the spectra according to the radial velocity of the stars. Defaults to False.
            save (bool, optional):
                If True save the list of stars to file. Defaults to True.
            filename (str, optional):
                If set, then read already fetched stars from the file
            directory (str, optional):
                Directory containing the file. Defaults to WORK_DIR defined in config file
        """
        if filename is not None:
            log.info(f'Reading the star list from file {os.path.join(directory, filename)}')
            self.starlist = StarsList(filename=filename, dir=directory)
            return

        self.starlist = StarsList(ra=self.ra, dec=self.dec, radius=self.radius)
        self.starlist.generate_gaia(self.wcs, gmag_limit=gmag_limit, shift=shift)

        if save:
            if not os.path.isdir(directory):
                os.mkdir(directory)
            outname = os.path.join(directory, f'{self.name}_starlist.fits.gz')
            log.info(f'Saving star list to: {outname}')
            self.starlist.save_to_fits(outname=outname)

    def generate_single_stars(self, parameters={}, shift=False, save=False):
        """
        Generate a list of stars by manually providing the basic informations

        Args:
            shift (bool, optional):
                apply the shift due to the velocity of the stars if available. Defaults to False.
            save (bool, optional):
                Save the starlist to file. Defaults to False.
            parameters (dictionary, optional):
                Dictionary containing the parameters to pass to the StarsList.add_star() method.
                Since more than one star can be generated simultaneously with this method, each
                type of parameter must be provided as lists. If no dictionary is provided, a
                single star is created with standard parameters.
                Defaults to {}.
        """

        self.starlist = StarsList(ra=self.ra, dec=self.dec, radius=self.radius)
        self.starlist.generate_stars_manually(self.wcs, parameters, shift=shift)

        if save:
            self.starlist.save_to_fits(outname=f'{self.name}_starlist.fits.gz')

    def open_starlist(self, filename, directory=lvmdatasimulator.WORK_DIR):
        """
        Open an existing file list from a fits file.

        Args:
            filename (str):
                Name of the fits file containing the StarList.
            directory (str, optional):
                Directory containing the file. Defaults to WORK_DIR defined in config file
        """

        self.starlist = StarsList(filename=filename, dir=directory)

    def show(self, obs_coords=None, subplots_kw=None, scatter_kw=None, fibers=None, outname=None,
             cmap=plt.cm.Oranges, percentile=98.):
        """
        Display the LVM field with overlaid apertures (if needed). This is a work in progress.

        Args:
            obs_coords (SkyCoord, optional):
                Coordinates of the center of the fiber bundle
            subplots_kw (dict, optional):
                keyword arguments to be passed to plt.subplots. Defaults to None.
            scatter_kw (dict, optional):
                keyword arguments to be passed to plt.scatter. Defaults to None.
            fibers (dict or list of fibers, optional):
                structure defining the position and size of aperture(s) for spectra extraction.
            outname (str, optional):
                name of the file where the output image will be saved (abs. path or relative to WORK_DIR or cur. dir)
            cmap (plt.cm, optional):
                colormap to use for imshow
            percentile (float, optional):
                percentile to use in ImageNormalize
        """

        if obs_coords is None:
            obs_coords = self.coord
            log.warning('Bundle center coords are not defined, using the coords of the field.')

        fig, ax = plt.subplots(1, 1, subplot_kw=dict(projection=self.wcs))
        if self.ism_map is None:
            self._get_ism_map(wavelength=6562.81)
        norm = ImageNormalize(self.ism_map, interval=PercentileInterval(percentile=percentile),
                              stretch=AsinhStretch())
        img = ax.imshow(self.ism_map, norm=norm, cmap=cmap)
        if self.starlist is not None:
            self._plot_stars(ax=ax)
        cb = plt.colorbar(img)
        cb.set_label(r"F(H$\alpha$), erg s$^{-1}$ cm$^{-2}$", fontsize=15)
        ax.set_xlabel('RA (J2000)', fontsize=15)
        ax.set_ylabel('Dec (J2000)', fontsize=15)

        if fibers is not None:
            for fiber in fibers:
                fiber_coord = obs_coords.spherical_offsets_by(fiber.x, fiber.y)
                p = SphericalCircle((fiber_coord.ra, fiber_coord.dec), fiber.diameter / 2,
                                    edgecolor='green', facecolor='none',
                                    transform=ax.get_transform('fk5'))
                ax.add_patch(p)
        if outname is not None:
            if outname.startswith("/") or outname.startswith("\\") or outname.startswith("."):
                cur_outname = outname
            else:
                cur_outname = os.path.join(lvmdatasimulator.WORK_DIR, outname)
            plt.savefig(cur_outname, dpi=200)
        plt.show()

    def _plot_stars(self, ax):
        """
        Create a scatter plot showing the position of the stars. This is just a preliminary
        thing.

        Args:
            ax (axis):
                axis where to plot the position of the stars.
        """
        if len(self.starlist.stars_table) > 0:
            ax.scatter(self.starlist.stars_table['ra'], self.starlist.stars_table['dec'],
                       transform=ax.get_transform('world'),
                       marker=(5, 1), c='r')
        pass

    def _create_ism(self, distance=50 * u.kpc, spec_resolution=0.06 * u.Angstrom,
                    sys_velocity=0 * velunit, turbulent_sigma=10. * velunit, **_):
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

    def get_map(self, wavelength_ranges=None, unit_range=u.AA):
        """
        Create the input map of the field in the selected wavelength range
        :param wavelength_ranges: exact wavelength or minimal and maximal wavelength for the map extraction
        :param unit_range: unit of the wavelength (default is angstrom)
        """

        # default value is Ha
        if wavelength_ranges is None:
            wavelength_ranges = [6562.81]

        if isinstance(wavelength_ranges[0], (float, int)) and len(wavelength_ranges) == 1:
            wavelength_ranges = [[wavelength_ranges[0] - 0.01, wavelength_ranges[0] + 0.01]]
        elif isinstance(wavelength_ranges[0], (float, int)):
            wavelength_ranges = [wavelength_ranges]

        for wavelength_range in wavelength_ranges:

            self.ism_map = np.zeros(shape=(self.npixels, self.npixels), dtype=float)
            for i in range(len(wavelength_range)):
                wavelength_range[i] *= unit_range
                wavelength_range[i] = wavelength_range[i].to(u.AA).value

            wavelength_range = np.atleast_1d(wavelength_range)
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

            save_file = f'{self.name}_{int(wavelength_range[0])}_{int(wavelength_range[1])}_input_map.fits'
            cur_save_file = os.path.join(lvmdatasimulator.WORK_DIR, save_file)
            header = self.wcs.to_header()
            header['LAMRANGE'] = ("{0}-{1}".format(wavelength_range[0], wavelength_range[1]),
                                  "Wavelength range used for image extraction")
            fits.writeto(cur_save_file, data=self.ism_map, header=header, overwrite=True)
            log.info("Input image in {0}-{1}AA wavelength range "
                     "is saved to {2}".format(wavelength_range[0], wavelength_range[1], save_file))

    def add_nebulae(self, list_of_nebulae=None, load_from_file=None, save_nebulae=None, overwrite=True):
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
                                'continuum_type': 'BB' or 'Model' or 'Poly' # type of the continuum model
                                'continuum_data': model_id or [poly_coefficients] or Teff # value defining cont. shape
                                'continuum_flux': 1e-16 * u.erg / u.cm ** 2 / u.s / u.arcsec **2 / u.AA,
                                'continuum_mag': 22 * u.mag,
                                'continuum_wl': 5500, # could be also R, V, B,
                                'ext_law': 'F99',  # Extinction law, one of those used by pyneb (used for dark nebulae)
                                'ext_rv': 3.1,  # Value of R_V for extinction curve calculation (used for dark nebulae)
                                }]
            load_from_file: path (asbolute or relative to work_dir) to the file with previously
                            calculated ISM part (preferred if both load_from_file and
                            list_of_nebulae are present)
            save_nebulae: path (asbolute or relative to WORK_DIR or current dir) where to save fits with
                          calculated ISM (only used if list_of_nebulae is present and
                          load_from_file is not)
            overwrite: if True (default) then the ISM content will be overwritten,
                       otherwise new nebulae will be added on top of the already generated

        """
        loaded = False
        if load_from_file is not None:
            cfile = load_from_file
            if not os.path.isfile(cfile):
                cfile = os.path.join(lvmdatasimulator.WORK_DIR, load_from_file)
                if not os.path.isfile(cfile):
                    log.warning("Cannot find file with ISM content to load: {}".format(load_from_file))
                    cfile = None
        else:
            cfile = None
        if cfile is not None:
            if overwrite and (self.ism.content[0].header.get('Nobj') > 0):
                self.ism = self._create_ism(**self.ism_params)
            loaded = self.ism.load_nebulae(cfile)
            if loaded:
                log.info("Nebulae successfully loaded from file")
                if self.ism_map is not None:
                    self._get_ism_map()
                return

        if list_of_nebulae is not None:
            if overwrite and (self.ism.content[0].header.get('Nobj') > 0):
                self.ism = self._create_ism(**self.ism_params)
            loaded = self.ism.generate(list_of_nebulae)
            if loaded and self.ism_map is not None:
                self._get_ism_map()
            if loaded and save_nebulae is not None:
                if not (save_nebulae.startswith('/') or save_nebulae.startswith(r'\\') or save_nebulae.startswith('.')):
                    save_nebulae = os.path.join(lvmdatasimulator.WORK_DIR, save_nebulae)
                self.ism.save_ism(save_nebulae)
        if not loaded:
            log.warning("Cannot load the nebulae! Check input parameters.")
            return None

    def shift_nebula(self, nebula_id, offset=(0., 0.), units=(u.pixel, u.pixel), save=None):
        """
        Shift a nebula position in the FOV
        :param nebula_id: number of the target component in the ISM storage (starting from 0)
        :param offset: tuple or list, defining the offset along x or y axes
        :param units: units of the offsets (in astropy.units values; default are pixels)
        :param save: path (asbolute or relative to WORK_DIR or current dir) where to save the updated fits file
        """
        if (offset[0] == 0) and (offset[1] == 0):
            log.warning("Both offsets are equal to 0 for nebula id={0}".format(nebula_id))
            return
        if self.ism.content[0].header.get('Nobj') < (nebula_id + 1):
            log.warning("Requested nebula_id={0} is absent in the ISM content".format(nebula_id))
            return
        all_extensions = [hdu.header.get('EXTNAME') for hdu in self.ism.content
                          if (hdu.header.get('EXTNAME') is not None)
                          and ("COMP_{0}".format(nebula_id) in hdu.header.get('EXTNAME'))]
        if len(all_extensions) == 0:
            log.warning("Requested nebula_id={0} is absent in the ISM content".format(nebula_id))
            return
        if self.ism.content[all_extensions[0]].header.get('Nebtype') == 'DIG':
            log.warning("DIG cannot be shifted")
            return
        pix_offsets = np.array([0, 0])
        xyname=['X', 'Y']
        for ind in range(2):
            if units[ind] is u.pixel:
                pix_offsets[ind] = np.round(offset[ind]).astype(int)
            else:
                try:
                    pix_offsets[ind] = np.round((offset[ind] * units[ind]).to(u.arcsec) / self.spaxel).astype(int)
                except Exception as e:
                    log.error("Exception raised during the shifting of the nebula_id={0} "
                              "along the {1} axis: {2}".format(nebula_id, xyname[ind], e))
                    return
        for cur_ext in all_extensions:
            self.ism.content[cur_ext].header['X0'] += pix_offsets[0]
            self.ism.content[cur_ext].header['Y0'] += pix_offsets[1]
        self._get_ism_map()
        if save is not None:
            if not (save.startswith('/') or save.startswith(r'\\') or save.startswith('.')):
                save = os.path.join(lvmdatasimulator.WORK_DIR, save)
            self.ism.save_ism(save)
        return

    def _get_ism_map(self, wavelength=6562.81):
        """
        Create map of ISM part in desired line or wavelength range (default = Halpha)
        """
        if self.ism.content[0].header['Nobj'] > 0:
            self.ism_map = self.ism.get_map(wavelength=wavelength)
        else:
            self.ism_map = np.zeros(shape=(self.npixels, self.npixels), dtype=float)

    def extract_spectra(self, fibers, wl_grid, obs_coords=None):
        """
        Perform spectra extraction within the given aperture.

        Args:
            fibers (list or tuple fiber objects):
                structure defining the position and size of aperture for spectra extraction
            wl_grid (numpy.array):
                wavelength grid for the resulting spectrum
            obs_coords (SkyCoord, optional):
                Coordinates of the center of the fiber bundle
        """

        if obs_coords is None:
            obs_coords = self.coord
            log.warning('Bundle center coords are not defined, using the coords of the field.')

        spectrum = np.zeros(shape=(len(fibers), len(wl_grid))) * fluxunit * u.arcsec ** 2
        fiber_id = []
        dl = (wl_grid[1] - wl_grid[0]).to(u.AA)
        xx, yy = np.meshgrid(np.arange(self.npixels), np.arange(self.npixels))
        aperture_mask = np.zeros(shape=(self.npixels, self.npixels), dtype=int)
        fibers_coords = np.zeros(shape=(len(fibers), 3), dtype=float)
        for index, fiber in enumerate(fibers):
            fiber_id.append(fiber.id)
            # I have to check this holds
            cur_fiber_coord = obs_coords.spherical_offsets_by(fiber.x, fiber.y)
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

            if self.starlist is not None and len(self.starlist.stars_table) > 0:
                # xc_stars = np.round(self.starlist.stars_table['x']).astype(int)
                # yc_stars = np.round(self.starlist.stars_table['y']).astype(int)
                # stars_id = np.flatnonzero(aperture_mask[yc_stars, xc_stars] == (index + 1))
                stars_id = np.flatnonzero((self.starlist.stars_table['x'] - xc) ** 2 +
                                          (self.starlist.stars_table['y'] - yc) ** 2 <= (s ** 2))
                for star_id in stars_id:
                    p = interp1d(self.starlist.wave.to(u.AA).value, self.starlist.spectra[star_id], bounds_error=False,
                                 fill_value='extrapolate')
                    # !!! APPLY EXTINCTION BY DARK NEBULAE TO STARS !!!
                    spectrum[index, :] += (p(wl_grid.value) * dl.value * fluxunit * u.arcsec ** 2)
        if np.max(aperture_mask) < fibers_coords.shape[0]:
            fibers_coords = fibers_coords[:np.max(aperture_mask), :]
        log.info("Start extracting nebular spectra")
        spectrum_ism = self.ism.get_spectrum(wl_grid.to(u.AA), aperture_mask, fibers_coords,
                                             self.spaxel.value)
        if spectrum_ism is not None:
            spectrum[: len(spectrum_ism), :] += spectrum_ism
        return np.array(fiber_id), spectrum / dl.value / u.AA
