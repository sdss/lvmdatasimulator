# encoding: utf-8
#
# @Author: Oleg Egorov, Enrico Congiu
# @Date: Nov 12, 2021
# @Filename: field.py
# @License: BSD 3-Clause
# @Copyright: Oleg Egorov, Enrico Congiu

import sys
if (sys.version_info[0]+sys.version_info[1]/10.) < 3.8:
    from backports.cached_property import cached_property
else:
    from functools import cached_property
import os.path

import astropy.units as u
import numpy as np

from dataclasses import dataclass
from astropy.time import Time
from astropy.coordinates import get_body, EarthLocation, AltAz, SkyCoord
from astroplan import moon_illumination

from lvmdatasimulator import WORK_DIR

import matplotlib.pyplot as plt
from astropy.visualization import astropy_mpl_style, quantity_support
plt.style.use(astropy_mpl_style)
quantity_support()


@dataclass
class Observation:
    """
    This class contains the principal informations on the observations to be simulated.

    Parameters:
        name (str):
            Name of the field. Defaults to 'LVM_field'
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

    name: str = 'LVM_field'
    ra: u.deg = 0.0 * u.deg
    dec: u.deg = 0.0 * u.deg
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

    @cached_property
    def localtime(self):
        return self.time + self.utcoffset

    @cached_property
    def target_coords(self):
        '''get target coordinates'''
        return SkyCoord(self.ra, self.dec)

    @cached_property
    def target_coords_altaz(self):
        '''get target altazimuthal coordinates'''
        return self.target_coords.transform_to(self._altaz)

    @cached_property
    def _altaz(self):
        return AltAz(obstime=self.time, location=self.location)

    @cached_property
    def total_time(self):
        '''get total exposure time for the observations'''
        return self.nexp * self.exptime

    @cached_property
    def moon_coords(self):
        '''get moon coordinates'''
        coord = get_body('moon', time=self.time, location=self.location)
        return coord

    @cached_property
    def sun_coords(self):
        '''get sun coordinates'''
        coord = get_body('sun', time=self.time, location=self.location)
        return coord

    @cached_property
    def moon_coords_altaz(self):
        '''get moon altazimuthal coordinates'''
        coord = self.moon_coords.transform_to(self._altaz)
        return coord

    @cached_property
    def sun_coords_altaz(self):
        '''get sun altazimuthal coordinates'''
        coord = self.sun_coords.transform_to(self._altaz)
        return coord

    @cached_property
    def mjd(self):
        '''get modified julian date'''
        return self.time.mjd

    @cached_property
    def jd(self):
        '''get julian date'''
        return self.time.jd

    @cached_property
    def moon_distance(self):
        '''get distance between the target field and the moon'''
        # this is weird. moon to target is ok, target to moon is not
        return self.moon_coords.separation(self.target_coords)

    @cached_property
    def moon_illumination(self):
        '''get moon illumination'''
        return round(moon_illumination(self.time), 3)

    @cached_property
    def days_from_new_moon(self):
        '''This is greatly approximated'''
        conversion = np.array([0, 0.01, 0.05, 0.11, 0.19, 0.27, 0.36, 0.46, 0.55,
                               0.65, 0.73, 0.81, 0.88, 0.93, 1])
        diff = np.abs(conversion - self.moon_illumination)
        return np.argmin(diff)

    @cached_property
    def airmass(self):
        '''get airmass of target from coordinates'''
        return self.target_coords_altaz.secz.value

    def plot_visibilities(self, dir=WORK_DIR, show=False):

        # preparing the plot for the full day
        delta_midnight = np.linspace(-12, 12, 100) * u.hour
        midnight = Time(self.time.value[:10] + 'T00:00:00', format='isot', scale='utc')
        times = midnight + delta_midnight
        frames = AltAz(obstime=times, location=self.location)
        sun_altaz = self.sun_coords.transform_to(frames)
        moon_altaz = self.moon_coords.transform_to(frames)
        target_altaz = self.target_coords.transform_to(frames)

        print(moon_altaz.alt)
        print(delta_midnight)

        deltaobs = np.arange(0, self.total_time.value, 120) * u.s
        delta_time = self.time - midnight
        offset = delta_time.value * 24 * u.hour + deltaobs
        obstime = self.time + deltaobs
        obsframe = AltAz(obstime=obstime, location=self.location)
        obs_altaz = self.target_coords.transform_to(obsframe)

        fig, ax = plt.subplots(1, 1)
        ax.plot(delta_midnight, sun_altaz.alt, color='gold', label='Sun')
        # ax.plot(delta_midnight, moon_altaz.alt, color='silver', ls='--', label='Moon')
        ax.plot(delta_midnight, target_altaz.alt, color='red', ls=':', label=self.name)
        ax.scatter(offset, obs_altaz.alt, lw=2, s=8, color='red', zorder=100)
        ax.fill_between(delta_midnight, 0 * u.deg, 90 * u.deg,
                        sun_altaz.alt < -0 * u.deg, color='0.5', alpha=0.5, zorder=0)
        ax.fill_between(delta_midnight, 0 * u.deg, 90 * u.deg,
                        sun_altaz.alt < -18 * u.deg, color='k', alpha=0.5, zorder=0)
        ax.legend(loc='upper left')
        ax.set_xlim(-12 * u.hour, 12 * u.hour)
        ax.set_xticks((np.arange(13) * 2 - 12) * u.hour)
        ax.set_ylim(0 * u.deg, 90 * u.deg)
        ax.set_xlabel('Hours from UT Midnight')
        ax.set_ylabel('Altitude [deg]')
        fig.savefig(os.path.join(dir,r'{self.name}_visibility.png'))
        if show:
            plt.show()
        else:
            plt.close()
