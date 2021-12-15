# encoding: utf-8
#
# @Author: Oleg Egorov, Enrico Congiu
# @Date: Nov 15, 2021
# @Filename: field.py
# @License: BSD 3-Clause
# @Copyright: Oleg Egorov, Enrico Congiu

import astropy.units as u
import numpy as np
import pyphot
import progressbar
# import matplotlib.pyplot as plt

# from pyphot import unit
from astropy.io import fits
from astropy.coordinates import SkyCoord
from astropy.table import Table, vstack
from astroquery.gaia import Gaia
from spectres import spectres
# from scipy.interpolate import interp1d

from lvmdatasimulator import log, ROOT_DIR

import os

# config parameters
Gaia.MAIN_GAIA_TABLE = "gaiadr2.gaia_source"  # Select Data Release 2. EDR3 is missing temperatures
kms = u.km / u.s
c = 299792.458 * kms


class StarsList:
    """
    Container of the list of stars for the LVMField object.

    This class allows to create, modify and store the list of stars needed by the LVMField object
    to create the datacube that will be feed to the LVM simulator.

    The class can also open a previously saved fits file with the correct informations.

    If no filename is provided, the class is initiated with an empty table, and stars can be added
    manually or by quering gaia on a particolar field.

    Parameters:
        ra (float, optional):
            right ascension of the center of the field. This parameter is expected in degree.
            Defaults to 0.
        dec (float, optional):
            declination of the center of the field. This parameter is expected in degree.
            Defaults to 0.
        radius (float, optional):
            radius of the field to be searched in Gaia. Defaults to 1.
        filename (str, optional):
            name of the file to be opened. If it is not provided an empty object is created using
            the default options or the user provided ones. Defaults to None.
        dir (str, optional):
            directory where the file to open is located. Defaults to './'
        unit_ra (astropy.unit, optional):
            unit associated to the right ascension. Defaults to u.deg
        unit_dec (astropy.unit, optional):
            unit associated to the declination. Defaults to u.deg
        unit_radius (astropy.unit, optional):
            unit associated to the radius variable. Defaults to u.arcmin
        colnames (list, optional):
            list of column names to initiate the table containing the list of stars.
        types (list, optional):
            data type to be associated to each column in the table

    Attributes:
        TBU
        colnames (list):
            list of column names to initiate the table containing the list of stars.
        stars_table (astropy.table):
            table containing the list of stars and their parameters
    """

    def __init__(self, ra=0, dec=0, radius=1, filename=None, dir='./',
                 unit_ra=u.deg, unit_dec=u.deg, unit_radius=u.arcmin,
                 colnames=['star_id', 'ra', 'dec', 'phot_g_mean_mag', 'phot_bp_mean_mag',
                           'phot_rp_mean_mag', 'teff_val', 'a_g_val', 'e_bp_min_rp_val',
                           'radial_velocity', 'gaia', 'source_id'],
                 types=[int, float, float, float, float, float, float, float,
                        float, float, bool, int],
                 units=[None, u.deg, u.deg, u.mag, u.mag, u.mag, u.K, u.mag, u.mag, kms, None,
                        None]
                 ):

        # if a filename is given open the file, otherwise create an object with the default
        # or user provided details
        if filename is not None:
            filename = os.path.join(dir, filename)
            self._read_from_fits(filename)

        else:
            # create an empty table to contain the list of stars
            self.ra = ra
            self.dec = dec
            self.radius = radius

            # fix the unit of measurements
            if isinstance(self.ra, float):
                self.ra *= unit_ra
            if isinstance(self.ra, float):
                self.dec *= unit_dec
            if isinstance(self.ra, float):
                self.radius *= unit_radius

            self.center = SkyCoord(self.ra, self.dec)
            self.colnames = colnames
            self.colunits = units
            self.stars_table = Table(names=self.colnames, dtype=types, units=units)
            self.stars_table.add_index('star_id')
            self.wave = None  # empty for now
            self.spectra = None  # empty for now
            self.library = None  # empty for now

    def __len__(self):
        return len(self.stars_table)

    def add_star(self, ra, dec, gmag, teff, ag, v):
        """
        Manually add a single star to the list.

        Manually add a star to the list. All the parameters are specified by the user.
        Temperature and extinction are needed to associate a spectrum to the star in order to build
        the datacube. The G-band magnitude will be used to normalize the spectra.

        Parameters:
            ra (float):
                right ascension of the star in degrees.
            dec (float):
                declination of the star in degrees.
            gmag (float):
                gaia G band magnitude of the star
            teff (float):
                effective temperature of the star (it is used to look for the correct spectrum)
                in K.
            ag (float):
                extinction on the gaia G band.


        """

        # check if the star is within the simulated FOV

        self._check_inside(ra, dec)

        new_row = {'star_id': len(self.stars_table) + 1,
                   'ra': ra,
                   'dec': dec,
                   'phot_g_mean_mag': gmag,
                   'teff_val': teff,
                   'a_g_val': ag,
                   'radial_velocity': v,
                   'gaia': False,
                   }

        log.info('star {} with Teff {}, Gmag {} and velocity {} added at position ({} , {})'
                 .format(new_row['star_id'], new_row['teff_val'], new_row['phot_g_mean_mag'],
                         new_row['radial_velocity'], new_row['ra'], new_row['dec']))

        self.stars_table.add_row(new_row)

    def _check_inside(self, ra, dec):
        """
        Check if the manually added star falls within the required FOV

        Args:
            ra (float):
                ra of the manually added star.
            dec (float):
                dec of the manually added star.

        Raises:
            ValueError:
                raise an error if the star is outside the required FOV
        """

        star_coord = SkyCoord(ra, dec, unit=(u.deg, u.deg))

        sep = star_coord.separation(self.center)

        if sep > self.radius:
            raise ValueError('This star is outside the simulated field of view...')

    def add_gaia_stars(self, gmag_limit=17):
        """
        Add stars from the Gaia DR2 catalog to the stars list.

        Query the Gaia DR2 catalog, select the stars brighter than gmag_limit and add the result
        to the list of stars used to simulated the observed field.

        Parameters:
            gmag_limit (float, optional):
                Maximum magnitude for a star to be included in the list. Defaults to 17.
        """
        try:
            result = query_gaia(self.center, self.radius.to(u.deg))
        except TimeoutError:
            log.warning('GAIA DR2 server timed out. Continuing without gaia stars')
            return

        # select only the relevant columns
        colnames = [item for item in self.colnames if item not in ['gaia', 'star_id']]
        result = result[colnames]

        # apply a filter on the magnitude of the stars
        mask = result['phot_g_mean_mag'] < gmag_limit
        result = result[mask]

        if len(result) == 0:
            log.warning('All the stars have been rejected!')
        else:
            log.info('{} stars are fainter than {} and have been rejected'
                     .format(len(mask) - mask.sum(), gmag_limit))

        # adding the star_id
        idx = range(len(self) + 1, len(result) + 1)
        result['star_id'] = idx

        # setting the gaia flag on the table
        result['gaia'] = np.ones(len(result), dtype=bool)

        # finally saving the new table
        self.stars_table = vstack([self.stars_table, result])

    def associate_spectra(self, shift=False,
                          library=f'{ROOT_DIR}/data/pollux_resampled_v0.fits.gz'):
        """
        Associate a spectrum from a syntetic library to each one of the stars in the list.

        Each star is associated to the spectrum with the closest temperature, which is rescaled to
        roughly match the observed gaia magnitude.

        Parameters:
            shift (bool, optional):
                shift the spectra according to the radial velocity of the stars
            library (str, optional):
                path to the spectral library to use.
                Defaults to '{ROOT_DIR}/data/pollux_resampled_v0.fits.gz'.
        """

        log.info('Associating spectra to stars')

        self.library = os.path.split(library)[1]

        log.info(f'Using library {self.library}')

        self.wave = self._get_wavelength_array(library)
        self.spectra = np.zeros((len(self.stars_table), len(self.wave)))

        bar = progressbar.ProgressBar(max_value=len(self.stars_table)).start()
        for i, row in enumerate(self.stars_table):
            spectrum = get_spectrum(row['teff_val'], library)
            if shift and row['radial_velocity']:
                spectrum = shift_spectrum(self.wave, spectrum, row['radial_velocity'])

            self.spectra[i] = spectrum
            bar.update(i)

        bar.finish()

    def rescale_spectra(self):
        """
        This function rescales the synthetic spectra in order to match them to the gaia photometry.
        It works only with the G band of gaia DR2.

        """

        log.info(f'Rescaling {len(self.stars_table)} synthetic spectra.')

        passband = pyphot.get_library()['GaiaDR2_G']

        # convert gaia magnitudes to fluxes in erg etc
        # I could do directly with mag, but pyphot has some problems when working on a single
        # spectra with magnitudes

        gaia_fluxes = passband.Vega_zero_flux * \
            10**(-0.4 * self.stars_table['phot_g_mean_mag'].data)

        synth_flux = passband.get_flux(self.wave.value, self.spectra, axis=1)

        scale_factor = gaia_fluxes / synth_flux  # scale factor for the spectra

        # I don't understand why but it does not work by just multiplying.
        # I'm not sure I want to keep going with this package
        for i, factor in enumerate(scale_factor):
            self.spectra[i] = self.spectra[i] * factor

    @staticmethod
    def _get_wavelength_array(filename=f'{ROOT_DIR}/data/pollux_resampled_v0.fits.gz',
                              unit=u.AA):

        with fits.open(filename) as hdu:

            wave = hdu['WAVE'].data * unit

        return wave

    def apply_extinction(self):
        pass

    def compute_star_positions(self, wcs):
        """
        Converting world coordinates to pixel coordinates.

        It can be used to build, in the future, a datacube or a 2D map

        Args:
            wcs (astropy.wcs):
                wcs of the source field that should be produced.
        """

        log.info('Transforming world coordinates to pixel coordinates')
        x, y = wcs.all_world2pix(self.stars_table['ra'], self.stars_table['dec'], 0)

        self.stars_table['x'] = x
        self.stars_table['y'] = y
        self.stars_table['x'].unit = u.pix
        self.stars_table['y'].unit = u.pix

    def save_to_fits(self, outname='starlist.fits.gz', outdir='./', overwrite=True):
        """
        Save the StarList as a fits file.

        Parameters:
            outname (str, optional):
                name of the output file. Defaults to 'starlist.fits.gz'.
            outdir (str, optional):
                path to the output directory. Defaults to './'.
            overwrite (bool, optional):
                overwrite the file if it already exist. Defaults to True.
        """

        # confirming that outfile is a fits or a compressed fits file
        accepted_types = ('.fits', '.fits.gz')

        if not outname.endswith(accepted_types):
            outname += '.fits.gz'
            log.warning(f'the name of the output file has been updated to {outname}')

        primary = fits.PrimaryHDU()  # creating the primary hdu

        # adding extension names in the primary header
        primary.header['EXT1'] = 'TABLE'
        primary.header['EXT2'] = 'FLUX'
        primary.header['EXT3'] = 'WAVE'
        primary.header['LIBRARY'] = (self.library, 'Stellar library')

        # Add other info in the header
        primary.header['RA'] = (self.ra.to(u.deg).value,
                                'Right ascension of the center of the field (deg)')
        primary.header['DEC'] = (self.dec.to(u.deg).value,
                                 'Declination of the center of the field (deg)')
        primary.header['RADIUS'] = (self.radius.to(u.deg).value,
                                    'Radius of the field (deg)')

        table = fits.table_to_hdu(self.stars_table)  # creating the table extension
        table.header['EXTNAME'] = 'TABLE'  # add name to the extension

        spectra = fits.ImageHDU(data=self.spectra, name='FLUX')  # creating the fluxes extension
        wave = fits.ImageHDU(data=self.wave.value, name='WAVE')  # creating the wave extension

        hdul = fits.HDUList([primary, table, spectra, wave])

        filename = os.path.join(outdir, outname)
        if overwrite and os.path.isfile(filename):
            log.warning(f'The file {filename} already exist and it will be overwritten')

        hdul.writeto(filename, overwrite=overwrite)

    def _read_from_fits(self, filename, dir='./'):
        """
        Read a starlist object from a fits file.

        The structure of the file must be the one required by the save_to_fits method and it
        must contain all the informations required to build a StarList object.
        It should be used to only open data previously saved by the save_to_fits method

        Parameters:
            filename (str):
                name of the file to open. It must be a .fits or a .fits.gz file
            dir (str, optional):
                directory where the file is located. Defaults to './'.

        Raises:
            ValueError:
                raises a ValueError when the file is not a .fits or a .fits.gz file.
        """

        accepted_types = ('.fits', '.fits.gz')

        if not filename.endswith(accepted_types):
            raise ValueError('Only .fits or .fits.gz files are accepted')

        filename = os.path.join(dir, filename)

        # reading the file
        with fits.open(filename) as hdu:

            # opening the main extensions
            self.stars_table = Table.read(hdu['TABLE'])
            self.spectra = hdu['FLUX'].data
            self.wave = hdu['WAVE'].data * u.AA

            # recovering main info from primary header
            self.ra = hdu[0].header['RA'] * u.deg
            self.dec = hdu[0].header['DEC'] * u.deg
            self.radius = hdu[0].header['RADIUS'] * u.deg
            self.center = SkyCoord(self.ra, self.dec)

            # recovering info on the name of the columns and their units
            self.colnames = self.stars_table.colnames
            self.colunits = []
            for col in self.colnames:
                self.colunits.append(self.stars_table[col].unit)
            self.stars_table.add_index('star_id')

    def generate(self, wcs, gmag_limit=17, shift=False):
        """
        Generate the star list and associate the spectra automatically

        Args:
            gmag_limit (float, optional):
                Maximum magnitude for a star to be included in the list. Defaults to 17.
            shift (bool, optional):
                shift the spectra according to the radial velocity of the stars. Defaults to False.
        """
        self.add_gaia_stars(gmag_limit=gmag_limit)
        self.compute_star_positions(wcs)
        self.associate_spectra(shift=shift)
        self.rescale_spectra()

    def remove_star(self, id):
        """
        Remove a star with a specific star_id

        Args:
            id (int):
                star_id of the star to be removed
        """

        # checking if id exist
        if id not in self.stars_table['star_id']:
            log.warning(f'There is no star with star_id = {id}')
            return

        # if it existh remove the star
        log.info(f'Removing star (star_id: {id})')

        mask = self.stars_table['star_id'] == id  # mask identifying the correct star
        self.stars_table = self.stars_table[~mask].copy()  # select all the other stars
        # self.stars_table.remove_row(id)

        # if spectra where already assigned, remove also the spectrum
        if self.spectra is not None:
            self.spectra = self.spectra[~mask].copy()

        assert len(self.stars_table) == len(self.spectra), \
            'The star and spectrum where not removed correctly'

################################################################################


def get_spectrum(temp, library):
    """
    Extract a spectrum from a provided library.

    The library should have at least two extensions, one called TEMP which includes the physical
    properties of the associated spectrum, and one called FLUX which contains an array where each
    column is a spectrum.


    Args:
        temp (float):
            temperature of the star for which the spectrum is needed
        library (str):
            path to the desired stellar spectral library

    Returns:
        array:
            simulated stellar spectrum with T ~ temp
    """

    with fits.open(library) as hdu:

        properties = Table.read(hdu['TEMP'])
        fluxes = hdu['FLUX'].data

    delta = np.abs(properties['T'] - temp)
    idx = np.argmin(delta)

    spectrum = fluxes[idx]
    return spectrum


def query_gaia(coord, radius):
    """
    Query Gaia DR2 catalog for sources in a given field.

    Query the Gaia DR2 catalog around a position given by 'coord' and a radius given by radius.
    Only the columns included in colnames are selected before returning the query.

    Parameters:
        coord (SkyCoord):
            coordinates of the field as a SkyCoord object.

        radius (Quantity):
            Radius of the field to be searched for around the central coordinates.

    Returns:
        astropy.Table:
            astropy table containing the result of the query.
    """

    job = Gaia.cone_search_async(coord, radius)
    results = job.get_results()

    if len(results) == 0:
        log.warning('No star detected!')
    else:
        log.info('{} Gaia stars in the field' .format(len(results)))

    return results


def shift_spectrum(wave, flux, radial_velocity, unit_v=kms):
    """
    Apply a shift to a spectrum based on the radial_velocity of the object.

    First corrects the wave and flux array, then resample the spectrum using the original
    wavelength range.

    Args:
        wave (array-like):
            array containing the wavelenght axis of the spectrum.
        flux (array-like):
            array with the fluxes
        radial_velocity (float):
            radial velocity of the object used for the shift
        unit_v (astropy.unit):
            unit of the radial_velocity

    Returns:
        array-like:
            resampled and shifted spectrum
    """

    radial_velocity *= unit_v
    z = radial_velocity / c
    new_wave = wave * (1 + z)
    new_flux = flux / (1 + z)

    resampled = spectres(wave.value, new_wave.value, new_flux)

    return resampled

# if __name__ == '__main__':

#     # # wave = np.arange(3000, 10000.1, 0.1)
#     # # open_gaia_passband(wave, band='G')

#     starlist = StarsList(0, 0, 2)
#     # starlist.add_gaia_stars(17)
#     # print(len(starlist))

#     starlist.add_star(0, 0, 7, 10000, 0.4)
#     # starlist.add_star(0, 0, 15, 20000, 0.4)
#     starlist.associate_spectra()
#     starlist.rescale_spectra()

#     # print(starlist.stars_table)
