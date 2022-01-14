# encoding: utf-8
#
# @Author: Oleg Egorov, Enrico Congiu
# @Date: Nov 12, 2021
# @Filename: field.py
# @License: BSD 3-Clause
# @Copyright: Oleg Egorov, Enrico Congiu

import functools
import astropy.units as u

from dataclasses import dataclass
from astropy.time import Time
from astropy.coordinates import get_body, EarthLocation, AltAz


@dataclass
class Observation:
    """
    This class contains the principal informations on the observations to be simulated.

    Parameters:
        time (str, optional):
            date and time of the observations in the following format:
            'YYYY-MM-DDTHH:MM:SS.SS'. Defaults to '2022-01-01T00:00:00.00'.
        location (EarthLocation, optional):
            location of the observatory to be assumed during the observations. Defaults to
            LCO.
        utcoffset (astropy.quantity, optional):
            offset of the location time with respect to UTC in hours. Defaults to -3 * u.hours
        exptime (astropy.quantity, optional):
            Exposure time of a single exposure in seconds. Defaults to 900s.
        nexp (int, optional):
            number of exposures to be acquired in this position. Defaults to 1.
        seeing (astropy.quantity, optional):
            seeing (in arcsec) at zenith during observations in the V band (5550 A).
            Defaults to 1 arcsec.
        sky_transparency (str, optional):
            sky transparency during the observations. Only three values available: 'PHOT', 'CLR',
            'THIN'. Defaults to 'PHOT'.

    Attributes:
        total_time (float):
            total exposure time of the observations. It's the results of nexp * exptime.
        moon_coords (skycoords):
            GCRS coordinates of the moon at the time of the observations
        sun_coords (skycoords):
            GCRS coordinates of the sun at the time of the observations
        moon_coords_altaz (skycoords):
            Alt-aziumuthal coordinates of the moon at the time of the observations.
        sun_coords_altaz (skycoords):
            Alt-aziumuthal coordinates of the sun at the time of the observations.
        mjd (float):
            mjd date of the observations
        jd (float):
            jd date of the observations


    Raises:
        ValueError: [description]

    """

    time: str = '2022-01-01T00:00:00.00'  # UT time of observations
    location: EarthLocation = EarthLocation.of_site('lco')
    utcoffset: u.hour = -3 * u.hour  # Correction to local time. It is important to keep it updated
    exptime: u.s = 900.0 * u.s  # exposure time in s
    nexp: int = 1  # number of exposures
    seeing: u.arcsec = 1 * u.arcsec  # seeing at zenit in the V-band (5500 A?)
    sky_transparency: str = 'PHOT'

    def __post_init__(self):
        self.time = Time(self.time, format='isot', scale='utc')  # time of obs.
        if self.sky_transparency not in ['PHOT', 'CLR', 'THIN']:
            raise ValueError(f'{self.sky_transparency} is not accepted.')
        self.localtime = self.time + self.utcoffset

    @functools.cached_property
    def total_time(self):
        return self.nexp * self.exptime

    @functools.cached_property
    def moon_coords(self):
        coord = get_body('moon', time=self.time, location=self.location)
        return coord

    @functools.cached_property
    def sun_coords(self):
        coord = get_body('sun', time=self.time, location=self.location)
        return coord

    @functools.cached_property
    def moon_coords_altaz(self):
        altaz = AltAz(obstime=self.time, location=self.location)
        coord = self.moon_coords.transform_to(altaz)
        return coord

    @functools.cached_property
    def sun_coords_altaz(self):
        altaz = AltAz(obstime=self.time, location=self.location)
        coord = self.sun_coords.transform_to(altaz)
        return coord

    @functools.cached_property
    def mjd(self):
        return self.time.mjd

    @functools.cached_property
    def jd(self):
        return self.time.jd