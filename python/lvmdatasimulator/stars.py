# encoding: utf-8
#
# @Author: Oleg Egorov, Enrico Congiu
# @Date: Nov 15, 2021
# @Filename: field.py
# @License: BSD 3-Clause
# @Copyright: Oleg Egorov, Enrico Congiu

import numpy as np
import astropy.units as u

from astropy.table import Table
from astropy.coordinates import SkyCoord
from astroquery.gaia import Gaia

from lvmdatasimulator import log

Gaia.MAIN_GAIA_TABLE = "gaiadr2.gaia_source"  # Select Data Release 2. EDR3 is missing temperatures


class StarsList:
    """Container of the list of stars for the LVMField object.

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
        """Manually add a single star to the list

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
               
        coords = SkyCoord(ra, dec, unit=(u.deg, u.deg))
        
        result = query_gaia(coords, radius)
        

def query_gaia(coord, radius, colnames):
    
    job = Gaia.cone_search(coord, radius)
    results = job.get_results()
    results = results[colnames]
    
    return results