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

from dataclasses import dataclass, field
from astropy.time import Time
from astropy.coordinates import get_body, EarthLocation, AltAz, SkyCoord
from astroplan import moon_illumination
from typing import List
from lvmdatasimulator import WORK_DIR

import matplotlib.pyplot as plt
from astropy.visualization import astropy_mpl_style, quantity_support
plt.style.use(astropy_mpl_style)
quantity_support()

import os

@dataclass
class Observation:
    """
    This class contains the principal informations on the observations to be simulated.

    Parameters:
        name (str, optional):
            Name of the field. Defaults to 'LVM_field'
        ra (float, optional):
            RA of the center of the fiber bundle. Defaults to 0.
        dec (float, optional):
            dec of the center of the fiber bundle. Defaults to 0.
        unit_ra (astropy.unit, optional):
            unit associated to the right ascension. Defaults to u.deg
        unit_dec (astropy.unit, optional):
            unit associated to the declination. Defaults to u.deg
        time (str, optional):
            date and time of the observations in the following format:
            'YYYY-MM-DDTHH:MM:SS.SS'. Defaults to '2022-01-01T00:00:00.00'.
        location (EarthLocation, optional):
            location of the observatory to be assumed during the observations. Defaults to
            LCO.
        utcoffset (astropy.quantity, optional):
            offset of the location time with respect to UTC in hours. Defaults to -3 * u.hours
        exptimes (astropy.quantity, optional):
            list of exposure time of a single exposure in seconds. Defaults to 900s.
        nexp (int, optional):
            number of exposures to be acquired in this position. Defaults to 1.
        seeing (astropy.quantity, optional):
            seeing (in arcsec) at zenith during observations in the V band (5550 A).
            Defaults to 1 arcsec.
        sky_transparency (str, optional):
            sky transparency during the observations. Only three values available: 'PHOT', 'CLR',
            'THIN'. Defaults to 'PHOT'.
        airmass (float, optional):
            Airmass of the target when observed. If set to None, the airmass will be automatically
            calculated from the target coordinates and the date and time of observetions.
            Defaults to None
        days_moon (int, optional):
            days from new monns at the moment of the observation. If set to None, it will be
            automatically calculated from the date and time of observetions. Defaults to None.
        sky_template (str, optional):
            full path to the sky models to use during the simulation. If set to None, the number
            of days from new_moon will be used to select one of the models for average conditions
            extracted from the ESO La Silla-Paranal sky model during the simulatin.
            Defaults to None


    Attributes:
        localtime (Time):
            local time at the observatory
        target_coords (SkyCoord):
            Equatorial coordinates of the target field.
        target_coords_altaz (SkyCoord):
            target coordinates in the Altazimuthal system.
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
        moon_distance (astropy.units):
            separation between the target and the moon.
        moon_illumination (float):
            return the moon illumination at a specific time.

    """

    name: str = 'LVM_field'
    ra: float = 0.0
    dec: float = 0.0
    unit_ra: u = u.deg
    unit_dec: u = u.deg
    time: str = '2022-01-01T00:00:00.00'  # UT time of observations
    location: EarthLocation = EarthLocation.of_site('lco')
    utcoffset: u.hour = -3 * u.hour  # Correction to local time. It is important to keep it updated
    exptimes: List[int] = field(default_factory=lambda: ['900'])  # exposure time in s
    nexp: int = 1  # number of exposures
    seeing: u.arcsec = 1 * u.arcsec  # seeing at zenit in the V-band (5500 A?)
    sky_transparency: str = 'PHOT'
    airmass: float = None
    days_moon: int = None
    sky_template: str = None
    geocoronal: float = None
    narcs: int = 1
    arc_exptimes: List[int] = field(default_factory=lambda: [10])
    flat_exptimes: List[int] = field(default_factory=lambda: [10])
    std_exptimes: List[int] = field(default_factory=lambda: [10])

    def __post_init__(self):
        # fix the unit of measurements
        if isinstance(self.ra, (float, int)):
            self.ra *= self.unit_ra
        if isinstance(self.dec, (float, int)):
            self.dec *= self.unit_dec
        if not isinstance(self.exptimes, list):
            self.exptimes = [self.exptimes]
        if not isinstance(self.arc_exptimes, list):
            self.arc_exptimes = [self.arc_exptimes]
        if not isinstance(self.flat_exptimes, list):
            self.flat_exptimes = [self.flat_exptimes]
        if not isinstance(self.std_exptimes, list):
            self.std_exptimes = [self.std_exptimes]

        self.time = Time(self.time, format='isot', scale='utc')  # time of obs.
        if self.sky_transparency not in ['PHOT', 'CLR', 'THIN']:
            raise ValueError(f'{self.sky_transparency} is not accepted.')

        if self.days_moon is None:
            self.days_moon = self.days_from_new_moon()
        elif self.days_moon < 0 or self.days_moon > 14:
            raise ValueError(f'days_moon must be between 0 and 14, while it is {self.days_moon}')

        if self.airmass is None:
            self.airmass = self.target_coords_altaz.secz.value
        elif self.airmass < 1:
            raise ValueError(f'airmass must be >= 1, but it is {self.airmass}')

    @cached_property
    def localtime(self):
        """ Return the local time based on the UT time and on the UTC offset"""
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
        """ Convert equatorial coordinates to altazimuthal"""
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
        return int(self.time.mjd)

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

    def days_from_new_moon(self):
        '''This is greatly approximated.
        Return the distance between the observations and the closest new moon in days.'''
        conversion = np.array([0, 0.01, 0.05, 0.11, 0.19, 0.27, 0.36, 0.46, 0.55,
                               0.65, 0.73, 0.81, 0.88, 0.93, 1])
        diff = np.abs(conversion - self.moon_illumination)
        return np.argmin(diff)

    def plot_visibilities(self, dir=WORK_DIR, show=False):
        """
        Plot the visibility of the target during the observations. Also the visibility of the moon
        and the sun during the day are plotted.

        Not sure if it is working at this stage

        Args:
            dir (str, optional):
                Directory where to save the output image. Defaults to WORK_DIR defined in the config file.
                In this case, Name will be also added to the path.
            show (bool, optional):
                If True, the plot is showed before saving. Otherwise, the plot is directly saved
                to file. Defaults to False.
        """

        # preparing the plot for the full day
        delta_midnight = np.linspace(-12, 12, 100) * u.hour
        midnight = Time(self.time.value[:10] + 'T00:00:00', format='isot', scale='utc')
        times = midnight + delta_midnight
        frames = AltAz(obstime=times, location=self.location)
        sun_altaz = self.sun_coords.transform_to(frames)
        moon_altaz = self.moon_coords.transform_to(frames)
        target_altaz = self.target_coords.transform_to(frames)

        # print(moon_altaz.alt)
        # print(delta_midnight)

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
        if dir == WORK_DIR:
            subdir = self.name
        else:
            subdir = ''
        fig.savefig(os.path.join(dir, subdir, r'{self.name}_visibility.png'))

        if show:
            plt.show()
        else:
            plt.close()
