# encoding: utf-8
#
# testing the code into the stars.py file

import astropy.units as u

from pytest import mark

from lvmdatasimulator.stars import StarsList


class TestStars:

    @mark.parametrize(('ra', 'dec', 'gmag', 'teff', 'ag'),
                      [(15, -25, 14, 8000, 0.4)])
    def test_add_star(self, ra, dec, gmag, teff, ag):
        """
        Test the add_star function
        """

        stars = StarsList()
        stars.add_star(ra, dec, gmag, teff, ag)

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

    @mark.parametrize(('ra', 'dec', 'radius', 'gmag_limit', 'nstars'),
                      [(0, 0, 2 * u.arcmin, 30, 10),
                       (0, 0, 2 * u.arcmin, 17, 2),
                       (0, 0, 2 * u.arcmin, 10, 0)])
    def test_add_gaia_stars(self, ra, dec, radius, gmag_limit, nstars):
        """
        Test the add_gaia_stars function
        """

        stars = StarsList()
        stars.add_gaia_stars(ra, dec, radius, gmag_limit=gmag_limit)
        assert len(stars) == nstars

    @mark.parametrize(('ra', 'dec', 'radius', 'gmag_limit'),
                      [(0, 0, 2 * u.arcmin, 10)])
    def test_add_gaia_stars_logging_rejected(self, ra, dec, radius, gmag_limit, caplog):
        """
        Test if the logging in the case of all stars being rejected is working
        """

        caplog.clear()
        stars = StarsList()
        stars.add_gaia_stars(ra, dec, radius, gmag_limit=gmag_limit)
        assert 'All the stars have been rejected!' in caplog.text

    @mark.parametrize(('ra', 'dec', 'radius', 'gmag_limit'),
                      [(0, 0, 2 * u.arcsec, 17)])
    def test_add_gaia_stars_logging_nostars(self, ra, dec, radius, gmag_limit, caplog):
        """
        Test if the logging works in the case there are no Gaia stars
        """

        caplog.clear()
        stars = StarsList()
        stars.add_gaia_stars(ra, dec, radius, gmag_limit=gmag_limit)
        assert 'No star detected!' in caplog.text
