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
# from spectres import spectres
# from scipy.interpolate import interp1d

from lvmdatasimulator import log, ROOT_DIR

import os

Gaia.MAIN_GAIA_TABLE = "gaiadr2.gaia_source"  # Select Data Release 2. EDR3 is missing temperatures


class StarsList:
    """
    Container of the list of stars for the LVMField object.

    This class allows to create, modify and store the list of stars needed by the LVMField object
    to create the datacube that will be feed to the LVM simulator.

    The class is initiated with an empty table, and stars can be added manually, or by quering gaia
    on a particolar field.

    Parameters:
        ra (float):
            right ascension of the center of the field. This parameter is expected in degree.
        dec (float):
            declination of the center of the field. This parameter is expected in degree.
        radius (float):
            radius of the field to be searched in Gaia.
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

    def __init__(self, ra, dec, radius,
                 unit_ra=u.deg, unit_dec=u.deg, unit_radius=u.arcmin,
                 colnames=['star_id', 'ra', 'dec', 'phot_g_mean_mag', 'phot_bp_mean_mag',
                           'phot_rp_mean_mag', 'teff_val', 'a_g_val', 'e_bp_min_rp_val',
                           'gaia', 'source_id'],
                 types=['int', 'float', 'float', 'float', 'float', 'float', 'float', 'float',
                        'float', 'bool', 'float'],
                 units=[None, u.deg, u.deg, u.mag, u.mag, u.mag, u.K, u.mag, u.mag, None, None]
                 ):

        # create an empty table to contain the list of stars
        self.ra = ra * unit_ra
        self.dec = dec * unit_dec
        self.center = SkyCoord(self.ra, self.dec)
        self.radius = radius * unit_radius
        self.colnames = colnames
        self.colunits = units
        self.stars_table = Table(names=self.colnames, dtype=types, units=units)
        self.stars_table.add_index('star_id')
        self.wave = None  # empty for now
        self.spectra = None  # empty for now

    def __len__(self):
        return len(self.stars_table)

    def add_star(self, ra, dec, gmag, teff, ag):
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

        new_row = {'star_id': len(self) + 1,
                   'ra': ra,
                   'dec': dec,
                   'phot_g_mean_mag': gmag,
                   'teff_val': teff,
                   'a_g_val': ag,
                   'gaia': False,
                   'source_id': np.nan}

        log.info('star {} with Teff {} and Gmag {} was added to star list at position ({} , {})'
                 .format(new_row['star_id'], new_row['teff_val'], new_row['phot_g_mean_mag'],
                         new_row['ra'], new_row['dec']))

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

    def associate_spectra(self, library=f'{ROOT_DIR}/data/pollux_resampled_v0.fits.gz'):
        """
        Associate a spectrum from a syntetic library to each one of the stars in the list.

        Each star is associated to the spectrum with the closest temperature, which is rescaled to
        roughly match the observed gaia magnitude.

        parameters:
            library (str, optional):
                path to the spectral library to use.
                Defaults to '{ROOT_DIR}/data/pollux_resampled_v0.fits.gz'.
        """

        log.info('Associating spectra to stars')

        self.wave = self._get_wavelength_array(library)
        self.spectra = np.zeros((len(self.stars_table), len(self.wave)))

        bar = progressbar.ProgressBar(max_value=len(self.stars_table)).start()
        for i, row in enumerate(self.stars_table):
            spectrum = get_spectrum(row['teff_val'], library)

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

    def save(self, outname='starlist.fits.gz', outdir='./', overwrite=True):
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

        table = fits.table_to_hdu(self.stars_table)  # creating the table extension
        spectra = fits.ImageHDU(data=self.spectra, name='FLUX')  # creating the fluxes extension
        wave = fits.ImageHDU(data=self.wave.value, name='WAVE')  # creating the wave extension

        hdul = fits.HDUList([primary, table, spectra, wave])

        filename = os.path.join(outdir, outname)
        if overwrite and os.path.isfile(filename):
            log.warning(f'The file {filename} already exist and it will be overwritten')

        hdul.writeto(filename, overwrite=overwrite)

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


# def open_gaia_passband(wave, band='G', conserve_flux=False):
#     """
#     Open the desired GAIA DR2 passband.

#     This function opens the desired GAIA DR2 passband and resample it on the same wavelength
#     of the spectra.

#     Args:
#         wave (1-D array-like object):
#             Array containing the wavelengths over which the passband must be resampled
#         band (str, optional):
#             GAIA DR2 passband to extract from the file. Defaults to 'G'. Other available bands
#             are 'BP' and 'RP'.
#         conserve_flux (bool, optional):
#             If True, it uses the spectres package to resample the passband. If False, it uses a
#             standard linear interpolation. Defaults to False.

#     Raises:
#         ValueError:
#             raised if the required passband is not included in the available ones

#     Returns:
#         numpy.array:
#             GAIA passband resampled on the wavelengths provided by the wave array
#     """

#     filename = '../../data/GaiaDR2_Passbands.dat'
#     available_bands = {'G': 'col2', 'BP': 'col4', 'RP': 'col6'}

#     # Only GAIA DR2 passbands are allowed
#     if band not in list(available_bands.keys()):
#         raise ValueError(f'The {band} band is not available. Try with G, BP or RP')

#     table = Table.read(filename, format='ascii')

#     old_wave = table['col1'] * 10  # from nm to A

#     sensitivity = table[available_bands[band]]  # can extract any of the three bands
#     mask = sensitivity == 99.99  # 99.99 is used to mark non existing data in the passband file
#     sensitivity[mask] = 0

#     if conserve_flux:
#         # using spectres: weird shape of spectrum
#         new_sensitivity = spectres(wave, old_wave, sensitivity)
#     else:
#         # interpolate the sensitivity to a better resolution
#         interpolated = interp1d(old_wave, sensitivity)
#         new_sensitivity = interpolated(wave)

#     # new_sensitivity = spectres(wave, old_wave, sensitivity)

#     # fig, ax = plt.subplots(1,1)

#     # ax.plot(wave, new_sensitivity, c='k', label='resampled')
#     # ax.plot(old_wave, sensitivity, c='r', label='original')
#     # ax.plot(wave, interp_sensitivity, c='b', label='scipy')
#     # ax.legend(loc='best')

#     # plt.show()

#     return new_sensitivity


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
