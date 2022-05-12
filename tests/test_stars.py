# encoding: utf-8
#
# testing the code into the stars.py file

import astropy.units as u
import pyphot
import numpy as np

from pytest import mark

from lvmdatasimulator.stars import StarsList


class TestStars:

    @mark.parametrize(('ra', 'dec', 'gmag', 'teff', 'ag', 'v'),
                      [(15, -25, 14, 8000, 0.4, 0)])
    def test_add_star(self, ra, dec, gmag, teff, ag, v):
        """
        Test the add_star function
        """

        stars = StarsList(15, -25, 3)
        stars.add_star(ra, dec, gmag, teff, ag, v)

        assert len(stars) == 1

        assert stars.stars_table.loc[1]['star_id'] == 1
        assert stars.stars_table.loc[1]['ra'] == ra
        assert stars.stars_table.loc[1]['dec'] == dec
        assert stars.stars_table.loc[1]['phot_g_mean_mag'] == gmag
        assert stars.stars_table.loc[1]['phot_bp_mean_mag'] == 0
        assert stars.stars_table.loc[1]['phot_rp_mean_mag'] == 0
        assert stars.stars_table.loc[1]['teff_val'] == teff
        assert stars.stars_table.loc[1]['a_g_val'] == ag
        assert not stars.stars_table.loc[1]['gaia']

    # @mark.parametrize(('ra', 'dec', 'radius', 'gmag_limit', 'nstars'),
    #                   [(0, 0, 2, 30, 10),
    #                    (0, 0, 2, 17, 2),
    #                    (0, 0, 2, 10, 0)])
    # def test_add_gaia_stars(self, ra, dec, radius, gmag_limit, nstars):
    #     """
    #     Test the add_gaia_stars function
    #     """

    #     stars = StarsList(ra, dec, radius)
    #     stars.add_gaia_stars(gmag_limit=gmag_limit)
    #     assert len(stars) == nstars

    # @mark.parametrize(('ra', 'dec', 'radius', 'gmag_limit'),
    #                   [(0, 0, 2, 10)])
    # def test_add_gaia_stars_logging_rejected(self, ra, dec, radius, gmag_limit, caplog):
    #     """
    #     Test if the logging in the case of all stars being rejected is working
    #     """

    #     caplog.clear()
    #     stars = StarsList(ra, dec, radius)
    #     stars.add_gaia_stars(gmag_limit=gmag_limit)
    #     assert 'All the stars have been rejected!' in caplog.text

    # @mark.parametrize(('ra', 'dec', 'radius', 'gmag_limit'),
    #                   [(0, 0, 2, 17)])
    # def test_add_gaia_stars_logging_nostars(self, ra, dec, radius, gmag_limit, caplog):
    #     """
    #     Test if the logging works in the case there are no Gaia stars
    #     """

    #     caplog.clear()
    #     stars = StarsList(ra, dec, radius, unit_radius=u.arcsec)
    #     stars.add_gaia_stars(gmag_limit=gmag_limit)
    #     assert 'No star detected!' in caplog.text

    # def test_associate_spectra(self):

    #     starlist = StarsList(0, 0, 2)
    #     starlist.add_star(0, 0, 7, 10000, 0.4)
    #     starlist.associate_spectra()

    # def test_rescale_spectra(self):
    #     """
    #     this test is a little bit janky
    #     """
    #     starlist = StarsList(0, 0, 2)
    #     starlist.add_star(0, 0, 7, 10000, 0.4)
    #     starlist.add_star(0, 0, 15, 15000, 0.4)
    #     starlist.associate_spectra()
    #     starlist.rescale_spectra()

    #     passband = pyphot.get_library()['GaiaDR2_G']
    #     print(starlist.spectra)
    #     mag = -2.5 * np.log10(passband.get_flux(starlist.wave.value, starlist.spectra, axis=1) /
    #                           passband.Vega_zero_flux)

    #     assert np.round(mag[0], 0) == starlist.stars_table['phot_g_mean_mag'][0]

    # def test_remove_star(self):

    #     starlist = StarsList(0, 0, 2)
    #     starlist.add_star(0, 0, 7, 10000, 0.4)
    #     starlist.add_star(0, 0, 15, 15000, 0.4)
    #     starlist.associate_spectra()

    #     len1 = len(starlist)

    #     starlist.remove_star(1)

    #     len2 = len(starlist)

    #     assert len1 == len2 + 1
    #     assert 1 not in starlist.stars_table['star_id']
