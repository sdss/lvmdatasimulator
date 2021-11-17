# encoding: utf-8
#
# @Author: Oleg Egorov, Enrico Congiu
# @Date: Nov 15, 2021
# @Filename: field.py
# @License: BSD 3-Clause
# @Copyright: Oleg Egorov, Enrico Congiu

import astropy.units as u
import numpy as np
import matplotlib.pyplot as plt

from astropy.coordinates import SkyCoord
from astropy.table import Table, vstack
from astroquery.gaia import Gaia
from spectres import spectres
from scipy.interpolate import interp1d

from lvmdatasimulator import log


Gaia.MAIN_GAIA_TABLE = "gaiadr2.gaia_source"  # Select Data Release 2. EDR3 is missing temperatures


class StarsList:
    """
    Container of the list of stars for the LVMField object.

    This class allows to create, modify and store the list of stars needed by the LVMField object
    to create the datacube that will be feed to the LVM simulator.

    The class is initiated with an empty table, and stars can be added manually, or by quering gaia
    on a particolar field.

    Parameters:
        colnames (list):
            list of column names to initiate the table containing the list of stars.

    Attributes:
        colnames (list):
            list of column names to initiate the table containing the list of stars.
        stars_table (astropy.table):
            table containing the list of stars and their parameters
    """

    def __init__(self,
                 colnames=['star_id', 'ra', 'dec', 'phot_g_mean_mag', 'phot_bp_mean_mag',
                           'phot_rp_mean_mag', 'teff_val', 'a_g_val', 'e_bp_min_rp_val',
                           'gaia', 'source_id'],
                 types=['int', 'float', 'float', 'float', 'float', 'float', 'float', 'float',
                        'float', 'bool', 'float']
                 ):

        # create an empty table to contain the list of stars
        self.colnames = colnames
        self.stars_table = Table(names=self.colnames, dtype=types)
        self.stars_table.add_index('star_id')

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
                right ascension of the star
            dec (float):
                declination of the star
            gmag (float):
                gaia G band magnitude of the star
            teff (float):
                effective temperature of the star (it is used to look for the correct spectrum)
            ag (float):
                extinction on the gaia G band

        """
        new_row = {'star_id': len(self) + 1,
                   'ra': ra,
                   'dec': dec,
                   'phot_g_mean_mag': gmag,
                   'teff_val': teff,
                   'a_g_val': ag,
                   'gaia': False,
                   'source_id': np.nan}

        log.info('star {} with Teff {} and Gmag {} was added to star list at position ({} , {}'
                 .format(new_row['star_id'], new_row['teff_val'], new_row['phot_g_mean_mag'],
                         new_row['ra'], new_row['dec']))

        self.stars_table.add_row(new_row)

    def add_gaia_stars(self, ra, dec, radius, gmag_limit=17):
        """
        Add stars from the Gaia DR2 catalog to the stars list.

        Query the Gaia DR2 catalog, select the stars brighter than gmag_limit and add the result
        to the list of stars used to simulated the observed field.

        Parameters:
            ra (float):
                right ascension of the center of the field. This parameter is expected in degree.
            dec (float):
                declination of the center of the field. This parameter is expected in degree.
            radius (Quantity):
                radius of the field to be searched in Gaia.
            gmag_limit (float, optional):
                Maximum magnitude for a star to be included in the list. Defaults to 17.

        """

        coords = SkyCoord(ra, dec, unit=(u.deg, u.deg))

        result = query_gaia(coords, radius)

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

    def associate_spectra(self):

        pass

    def rescale_spectra(self):

        pass


def open_gaia_passband(wave, band='G', conserve_flux=False):
    """
    Open the desired GAIA DR2 passband.

    This function opens the desired GAIA DR2 passband and resample it on the same wavelength
    of the spectra.

    Args:
        wave (1-D array-like object):
            Array containing the wavelengths over which the passband must be resampled
        band (str, optional):
            GAIA DR2 passband to extract from the file. Defaults to 'G'. Other available bands are
            'BP' and 'RP'.
        conserve_flux (bool, optional):
            If True, it uses the spectres package to resample the passband. If False, it uses a
            standard linear interpolation. Defaults to False.

    Raises:
        ValueError:
            raised if the required passband is not included in the available ones

    Returns:
        numpy.array:
            GAIA passband resampled on the wavelengths provided by the wave array
    """

    filename = '../../data/GaiaDR2_Passbands.dat'
    available_bands = {'G': 'col2', 'BP': 'col4', 'RP': 'col6'}

    # Only GAIA DR2 passbands are allowed
    if band not in list(available_bands.keys()):
        raise ValueError(f'The {band} band is not available. Try with G, BP, RP')

    table = Table.read(filename, format='ascii')

    old_wave = table['col1'] * 10  # from nm to A

    sensitivity = table[available_bands[band]]  # can extract any of the three bands
    mask = sensitivity == 99.99  # 99.99 is used to mark non existing data in the passband file
    sensitivity[mask] = 0

    if conserve_flux:
        # using spectres: weird shape of spectrum
        new_sensitivity = spectres(wave, old_wave, sensitivity)
    else:
        # interpolate the sensitivity to a better resolution
        interpolated = interp1d(old_wave, sensitivity)
        new_sensitivity = interpolated(wave)

    # new_sensitivity = spectres(wave, old_wave, sensitivity)

    # fig, ax = plt.subplots(1,1)

    # ax.plot(wave, new_sensitivity, c='k', label='resampled')
    # ax.plot(old_wave, sensitivity, c='r', label='original')
    # ax.plot(wave, interp_sensitivity, c='b', label='scipy')
    # ax.legend(loc='best')

    # plt.show()

    return new_sensitivity


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


if __name__ == '__main__':

    wave = np.arange(3000, 10000.1, 0.1)
    open_gaia_passband(wave, band='G')