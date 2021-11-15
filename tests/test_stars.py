# encoding: utf-8
#
# testing the code into the stars.py file

from pytest import mark

from lvmdatasimulator.stars import StarsList


class TestStars:

    @mark.parametrize(('ra', 'dec', 'gmag', 'teff', 'ag'),
                      [(15, -25, 14, 8000, 0.4)])
    def test_add_star(self, ra, dec, gmag, teff, ag):
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
        assert stars.stars_table.loc[1]['gaia'] == False

