# encoding: utf-8
#
# @Author: Oleg Egorov, Enrico Congiu
# @Date: Nov 15, 2021
# @Filename: ism.py
# @License: BSD 3-Clause
# @Copyright: Oleg Egorov, Enrico Congiu

# import scipy.optimize
# import sys

from astropy import units as u
from astropy import constants as c
import numpy as np
from astropy.io import fits
# from matplotlib import pyplot as plt
# from scipy.integrate import nquad
# import tqdm
# from multiprocessing import Pool
# import multiprocessing
from scipy.special import sph_harm
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales
from astropy.coordinates import SkyCoord
# from astropy.table import QTable
from dataclasses import dataclass
import functools
from scipy.ndimage.interpolation import map_coordinates
from scipy.interpolate import interp1d
# from typing import List
from lvmdatasimulator import log
from joblib import Parallel, delayed


fluxunit = u.erg / (u.cm ** 2 * u.s * u.arcsec ** 2)
velunit = u.km / u.s

def set_default_dict_values(mydict, key_to_check, default_value):
    if key_to_check not in mydict:
        mydict[key_to_check] = default_value


def brightness_inhomogeneities_sphere(harm_amplitudes, ll, phi_cur, theta_cur, rho, med, radius, thickness):
    brt = theta_cur * 0
    for m in np.arange(-ll, ll + 1):
        brt += (harm_amplitudes[m + ll * (ll + 1) - 1] * sph_harm(m, ll, phi_cur, theta_cur).real * med *
                (1 - np.sqrt(abs(rho.value ** 2 / radius.value ** 2 - (1 - thickness / 2) ** 2))))
    return brt


def limit_angle(value, bottom_limit=0, top_limit=np.pi):
    value[value < bottom_limit] += (top_limit-bottom_limit)
    value[value > top_limit] -= (top_limit - bottom_limit)
    return value


def xyz_to_sphere(x, y, z, pxscale=1.*u.pc):
    phi_c = np.arctan2(y, x)
    rad_c = (np.sqrt(x ** 2 + y ** 2 + z ** 2))
    rad_c[rad_c == 0 * u.pc] = 1e-3 * pxscale
    theta_c = (np.arccos(z / rad_c))
    phi_c = limit_angle(phi_c, 0 * u.radian, 2 * np.pi * u.radian)
    theta_c = limit_angle(theta_c, 0 * u.radian, np.pi * u.radian)
    return phi_c, theta_c, rad_c


@dataclass
class Nebula:
    """
    Base class defining properties of every nebula type.
    By itself it describes the rectangular nebula (for now - as DIG)
    """
    xc: int = None  # Center of the region in the field of view, pix
    yc: int = None  # Center of the region in the field of view, pix
    x0: int = 0  # Coordinates of the bottom-left corner in the field of view, pix
    y0: int = 0  # Coordinates of the bottom-left corner in the field of view, pix
    width: int = 101  # full width of cartesian grid, pix (should be odd)
    height: int = 101  # full height of cartesian grid, pix (should be odd)
    pxscale: u.pc = 0.01 * u.pc  # pixel size in pc
    spectrum_id: int = 0  # ID of a template Cloudy spectrum for this nebula
    sys_velocity: velunit = 0 * velunit  # Systemic velocity
    turbulent_sigma: velunit = 10 * velunit  # Velocity dispersion due to turbulence; included in calculations of LSF
    max_brightness: fluxunit = 1e-15 * fluxunit
    max_extinction: u.mag = 0 * u.mag
    n_process: int = 8  # maximal number of processes for parallelization
    _npix_los: int = 1  # full size along line of sight in pixels

    def __post_init__(self):
        if (self.xc is not None) and (self.yc is not None):
            self.x0 = self.xc - np.round((self.width - 1) / 2).astype(int)
            self.y0 = self.yc - np.round((self.height - 1) / 2).astype(int)
        elif (self.x0 is not None) and (self.y0 is not None):
            self.xc = self.x0 + np.round((self.width - 1) / 2).astype(int)
            self.yc = self.y0 + np.round((self.height - 1) / 2).astype(int)

    @functools.cached_property
    def _cartesian_x_grid(self):
        return np.arange(self.width) * self.pxscale

    @functools.cached_property
    def _cartesian_y_grid(self):
        return np.arange(self.height) * self.pxscale

    @functools.cached_property
    def _cartesian_z_grid(self):
        return np.arange(self._npix_los) * self.pxscale

    @functools.cached_property
    def _max_density(self):
        return self.max_extinction * (1.8e21 / (u.cm ** 2 * u.mag))

    @functools.cached_property
    def _brightness_3d_cartesian(self):
        """
        Method to obtain the brightness (or density) distribution of the nebula in cartesian coordinates
        """
        return np.ones(shape=(self.height, self.width, self._npix_los), dtype=float) / self._npix_los

    @functools.cached_property
    def brightness_skyplane(self):
        if self.max_brightness > 0:
            map2d = np.nansum(self._brightness_3d_cartesian, 2)
            return map2d / np.max(map2d) * self.max_brightness
        else:
            return None

    @functools.cached_property
    def extinction_skyplane(self):
        if self.max_extinction > 0:
            map2d = np.nansum(self._brightness_3d_cartesian, 2)
            return map2d / np.max(map2d) * self._max_density / (1.8e21 / (u.cm ** 2 * u.mag))
        else:
            return None

    @functools.cached_property
    def los_velocity(self):
        return np.atleast_1d(self.sys_velocity)

    @functools.cached_property
    def line_profile(self):
        lprf = np.zeros(shape=len(self.los_velocity), dtype=float)
        lprf[np.floor(len(lprf) / 2.).astype(int)] = 1.
        return lprf


@dataclass
class Filament(Nebula):
    """
    Class of an isotropic cylindrical shape filament.
    Defined by its position, lenght, PA, radius, maximal optical depth
    if it is emission-type filament, then maximal brightness
    """
    PA: u.degree = -40 * u.degree  # position angle of the filament
    length: u.pc = 10 * u.pc  # full length of the filament
    width: u.pc = 0.1 * u.pc  # full width (diameter) of the filament
    vel_gradient: (velunit / u.pc) = 0  # velocity gradient along the filament (to be added)
    _theta_bins: int = 50
    _rad_bins: int = 10
    _h_bins: int = 2
    _npix_los: int = 101

    def __post_init__(self):
        if (self.xc is not None) and (self.yc is not None):
            self.x0 = self.xc - np.round((len(self._cartesian_y_grid) - 1) / 2).astype(int)
            self.y0 = self.yc - np.round((len(self._cartesian_z_grid) - 1) / 2).astype(int)
        elif (self.x0 is not None) and (self.y0 is not None):
            self.xc = self.x0 + np.round((len(self._cartesian_y_grid) - 1) / 2).astype(int)
            self.yc = self.y0 + np.round((len(self._cartesian_z_grid) - 1) / 2).astype(int)

    @functools.cached_property
    def _theta_grid(self):
        return np.linspace(0, 2 * np.pi, self._theta_bins)

    @functools.cached_property
    def _h_grid(self):
        return np.linspace(0, self.length, self._h_bins)

    @functools.cached_property
    def _rad_grid(self):
        return np.linspace(0, self.width / 2, self._rad_bins)

    @functools.cached_property
    def _cartesian_y_grid(self):
        npix = np.ceil(1.01 * (self.length * np.abs(np.sin(self.PA)) +
                               self.width * np.abs(np.cos(self.PA))) / self.pxscale).astype(int)
        npix_l = npix / 2 - np.ceil(self.length / 2 * np.sin(-self.PA) / self.pxscale).astype(int)
        return (np.linspace(0, npix, npix + 1) - npix_l) * self.pxscale

    @functools.cached_property
    def _cartesian_z_grid(self):
        npix = np.ceil(1.01 * (self.length * np.abs(np.cos(self.PA)) +
                               self.width * np.abs(np.sin(self.PA))) / self.pxscale).astype(int)
        npix_l = npix / 2 - np.ceil(self.length / 2 * np.cos(-self.PA) / self.pxscale).astype(int)
        return (np.linspace(0, npix, npix + 1) - npix_l) * self.pxscale

    @functools.cached_property
    def _cartesian_x_grid(self):
        return np.linspace(-1.01, 1.01, self._npix_los) * self.width / 2

    @functools.cached_property
    def _brightness_3d_cylindrical(self):
        """
        Method to calculate brightness (or opacity) of the cloud at given theta, phi and radii

        theta: float -- azimuthal angle [0, 2 * np.pi]
        rad: float -- radius [0, self.width / 2]
        h: float -- height [0, self.length]
        Returns:
            3D cube of normalized brightness in theta-rad-h grid; total brightness = 1
        """
        rho, theta, h = np.meshgrid(self._rad_grid, self._theta_grid, self._h_grid, indexing='ij')
        brt = np.ones_like(theta)
        brt[rho > (self.width / 2)] = 0
        norm = np.sum(brt)
        if norm > 0:
            brt = brt / np.sum(brt)
        return brt

    @functools.cached_property
    def _brightness_3d_cartesian(self):
        x, y, z = np.meshgrid(self._cartesian_x_grid, self._cartesian_y_grid,
                              self._cartesian_z_grid, indexing='ij')

        h_c = -y * np.sin(self.PA) + z * np.cos(self.PA)
        theta_c = np.arctan2(y * np.cos(self.PA) + z * np.sin(self.PA), x)
        rad_c = np.sqrt(x ** 2 + (y * np.cos(self.PA) + z * np.sin(self.PA)) ** 2)
        rad_c[rad_c == 0 * u.pc] = 1e-3 * self.pxscale

        theta_c = limit_angle(theta_c, 0 * u.radian, 2 * np.pi * u.radian)

        ir = interp1d(self._rad_grid, np.arange(self._rad_bins), bounds_error=False)
        ith = interp1d(self._theta_grid, np.arange(self._theta_bins))
        ih = interp1d(self._h_grid, np.arange(self._h_bins), bounds_error=False)
        new_ir = ir(rad_c.ravel())
        new_ith = ith(theta_c.ravel())
        new_ih = ih(h_c.ravel())

        cart_data = map_coordinates(self._brightness_3d_cylindrical,
                                    np.vstack([new_ir, new_ith, new_ih]),
                                    order=1, mode='constant', cval=0)

        return cart_data.reshape([len(self._cartesian_x_grid),
                                  len(self._cartesian_y_grid),
                                  len(self._cartesian_z_grid)]).T


@dataclass
class DIG(Nebula):
    """
    Class defining the DIG component. For now it is defined just by its brightness (constant)
    """
    max_brightness: fluxunit = 1e-17 * fluxunit
    vel_gradient: (velunit / u.pc) = 0


@dataclass
class Cloud(Nebula):
    """Class of an isotropic spherical gas cloud without any ionization source.
    Defined by its position, radius, density, maximal optical depth"""
    radius: u.pc = 1.0 * u.pc
    max_brightness: fluxunit = 0 * fluxunit
    max_extinction: u.mag = 2.0 * u.mag
    thickness: float = 1.0
    harm_maxdegree: int = 1
    harm_amplitude: float = 0.1
    _phi_bins: int = 180
    _theta_bins: int = 180
    _rad_bins: int = 100
    _npix_los: int = 100

    def __post_init__(self):
        delta = np.round((len(self._cartesian_y_grid) - 1) / 2).astype(int)
        if (self.xc is not None) and (self.yc is not None):
            self.x0 = self.xc - delta
            self.y0 = self.yc - delta
        elif (self.x0 is not None) and (self.y0 is not None):
            self.xc = self.x0 + delta
            self.yc = self.y0 + delta

    @functools.cached_property
    def _theta_grid(self):
        return np.linspace(0, np.pi, self._theta_bins)

    @functools.cached_property
    def _phi_grid(self):
        return np.linspace(0, 2 * np.pi, self._phi_bins)

    @functools.cached_property
    def _rad_grid(self):
        return np.linspace(0, self.radius, self._rad_bins)

    @functools.cached_property
    def _cartesian_z_grid(self):
        npix = np.ceil(1.02 * self.radius / self.pxscale).astype(int)
        return np.linspace(-npix, npix, 2 * npix + 1) * self.pxscale

    @functools.cached_property
    def _cartesian_y_grid(self):
        return self._cartesian_z_grid.copy()

    @functools.cached_property
    def _cartesian_x_grid(self):
        return np.linspace(-1.02, 1.02, self._npix_los) * self.radius

    @functools.cached_property
    def _brightness_3d_spherical(self):
        """
        Method to calculate brightness (or opacity) of the cloud at given theta, phi and radii

        theta: float -- polar angle [0, np.pi]
        phi: float -- azimuthal angle [0, 2 * np.pi]
        rad: float -- radius [0, self.radius]
        Returns:
            3D cube of normalized brightness in theta-phi-rad grid; total brightness = 1
        """
        rho, theta, phi = np.meshgrid(self._rad_grid, self._theta_grid, self._phi_grid, indexing='ij')
        brt = np.ones_like(theta)
        brt[rho < (self.radius * (1 - self.thickness))] = 0
        brt[rho > self.radius] = 0
        med = np.median(brt[brt > 0])

        if self.harm_maxdegree > 0:
            phi_cur = limit_angle(phi + np.random.uniform(0, 2 * np.pi, 1), 0, 2 * np.pi)
            theta_cur = limit_angle(theta + np.random.uniform(0, np.pi, 1), 0, np.pi)
            harm_amplitudes = self.harm_amplitude * np.random.randn(self.harm_maxdegree * (self.harm_maxdegree + 2))

            brt += np.sum(Parallel(n_jobs=self.n_process)(delayed(brightness_inhomogeneities_sphere)
                                                          (harm_amplitudes, ll, phi_cur, theta_cur,
                                                           rho, med, self.radius, self.thickness)
                                                          for ll in np.arange(1, self.harm_maxdegree+1)), axis=0)
            brt[brt < 0] = 0
        if med > 0:
            brt = brt / np.sum(brt)
        return brt

    def _interpolate_sphere_to_cartesian(self, spherical_array):
        x, y, z = np.meshgrid(self._cartesian_x_grid,
                              self._cartesian_y_grid,
                              self._cartesian_z_grid, indexing='ij')
        phi_c, theta_c, rad_c = xyz_to_sphere(x, y, z, pxscale=self.pxscale)
        ir = interp1d(self._rad_grid, np.arange(self._rad_bins), bounds_error=False)
        ith = interp1d(self._theta_grid, np.arange(self._theta_bins))
        iphi = interp1d(self._phi_grid, np.arange(self._phi_bins))
        new_ir = ir(rad_c.ravel())
        new_ith = ith(theta_c.ravel())
        new_iphi = iphi(phi_c.ravel())
        cart_data = map_coordinates(spherical_array,
                                    np.vstack([new_ir, new_ith, new_iphi]),
                                    order=1, mode='constant', cval=0)
        return cart_data.reshape([len(self._cartesian_x_grid),
                                  len(self._cartesian_y_grid),
                                  len(self._cartesian_z_grid)]).T

    @functools.cached_property
    def _brightness_3d_cartesian(self):
        return self._interpolate_sphere_to_cartesian(self._brightness_3d_spherical)


@dataclass
class Bubble(Cloud):
    """Class of an isotropic thin expanding bubble."""
    spectral_axis: velunit = np.arange(-20, 20, 10) * velunit
    expansion_velocity: velunit = 20 * velunit
    max_brightness: fluxunit = 1e-15 * fluxunit
    max_extinction: u.mag = 0 * u.mag
    thickness: float = 0.2

    @functools.cached_property
    def _velocity_3d_spherical(self) -> velunit:
        """
        Calculate line of sight velocity at given radius, phi, theta

        V ~ 1/brightness (given that v~1/n_e^2 and brightness~ne^2)
        """
        rho, theta, phi = np.meshgrid(self._rad_grid, self._theta_grid, self._phi_grid, indexing='ij')
        vel_cube = np.zeros_like(self._brightness_3d_spherical)
        rec = (rho <= self.radius) & (rho >= (self.radius * (1 - self.thickness)))
        vel_cube[rec] = np.sin(theta[rec]) * np.cos(phi[rec]) * self.expansion_velocity / \
            self._brightness_3d_spherical[rec] * \
            np.median(self._brightness_3d_spherical[self._brightness_3d_spherical > 0])
        return vel_cube

    @functools.cached_property
    def _velocity_3d_cartesian(self) -> velunit:
        return self._interpolate_sphere_to_cartesian(self._velocity_3d_spherical)

    def _turbulent_lsf(self, velocity):
        """Line spread function as a function of coorinates, including the velocity center shift"""
        # mu = self.velocity(theta, phi)
        mu = self._velocity_3d_cartesian[:, :, :, None] * velunit + self.sys_velocity
        sig = self.turbulent_sigma
        return 1. / (np.sqrt(2. * np.pi) * sig) * np.exp(-np.power((velocity - mu) / sig, 2.) / 2)

    def _d_spectrum_cartesian(self, velocity: velunit):
        """Returns local spectrum, per pc**3 of area"""
        return (self._brightness_3d_cartesian[:, :, :, None] * (fluxunit / u.pc ** 3)
                * self._turbulent_lsf(velocity)).to(fluxunit / velunit / u.pc ** 3)

    @functools.cached_property
    def vel_field(self) -> (fluxunit / velunit):
        """
        Produces the distribution of the observed line profiles in each pixels of the sky plane
        """
        vel_axis = self.spectral_axis.to(velunit, equivalencies=u.spectral())
        _, _, _, vels = np.meshgrid(self._cartesian_z_grid,
                                    self._cartesian_y_grid,
                                    self._cartesian_x_grid,
                                    vel_axis, indexing='ij')
        spectrum = (
                np.sum(self._d_spectrum_cartesian(vels), axis=2).T
                * (self._cartesian_x_grid[1] - self._cartesian_x_grid[0])
                * (self._cartesian_y_grid[1] - self._cartesian_y_grid[0])
                * (self._cartesian_z_grid[1] - self._cartesian_z_grid[0])
        )
        return spectrum / np.sum(spectrum, axis=0)


@dataclass
class ISM:
    """
    Class defining the ISM contribution to the field of view
    """
    wcs: WCS
    width: int = 400  # Width of field of view in pixels
    height: int = 400  # Width of field of view in pixels
    spec_resolution: u.Angstrom = 0.06 * u.Angstrom  # Spectral resolution of the simulation
    npix_line: int = 1  # Minimal number of pixels for a resolution element at wl = 10000A for construction of vel.grid
    distance: u.kpc = 50 * u.kpc  # Distance to the object for further conversion between arcsec and pc
    sys_velocity: velunit = 0 * velunit  # Systemic velocity to center the vel.grid on
    vel_amplitude: velunit = 100 * velunit  # Maximal deviation from the systemic velocity to setup vel.grid
    # nebulae: List = None#List[None]
    turbulent_sigma: velunit = 10. * velunit  # turbulence vel. disp. to be used for every nebula unless other specified
    last_id: int = 0
    ext_eps: u.mag = 0.01 * u.mag
    brt_eps: fluxunit = 1e-20 * fluxunit
    vel_contrib_eps: float = 1e-3

    def __post_init__(self):
        self.content = fits.HDUList()
        self.content.append(fits.PrimaryHDU())
        self.content[0].header['Nobj'] = 0  # number of generated nebulae
        # self.content = #QTable(names=["ObjID", 'X', 'Y', 'Brightness', 'Extinction',
        #                              'LineProfile', 'TurbVelocity', 'SpectrumID', 'Zorder'],
        #                       dtype=(int, int, int, float, float, np.ndarray, float, int, int),
        #                       units={'Brightness': fluxunit, 'Extinction': u.mag, 'TurbVelocity': velunit})
        self.vel_grid = np.linspace(-self.vel_amplitude + self.sys_velocity,
                                    self.vel_amplitude + self.sys_velocity,
                                    np.ceil(self.vel_amplitude / self.vel_resolution).astype(int) * 2 + 1)
        self.pxscale = proj_plane_pixel_scales(self.wcs)[0] * 3600 * self.distance.to(u.pc) / 206265.

    @functools.cached_property
    def vel_resolution(self):
        return (self.spec_resolution / self.npix_line / (10000 * u.Angstrom) * c.c).to(velunit)

    # def _add_pix_to_tab(self, params):
    #     x = params[0]
    #     y = params[1]
    #     obj_to_add = self._cur_obj_to_add
    #     turb_velocity = params[2]
    #     zorder = params[3]
    #     spectrum_id = params[4]
    #
    #     if obj_to_add.max_brightness > 0:
    #         cur_brt = obj_to_add.brightness_skyplane[x, y]
    #         if cur_brt < self.brt_eps:
    #             cur_brt = 0
    #     else:
    #         cur_brt = 0
    #     if obj_to_add.max_extinction > 0:
    #         cur_ext = obj_to_add.extinction[x, y]
    #         if cur_ext < self.ext_eps:
    #             cur_ext = 0
    #     else:
    #         cur_ext = 0
    #
    #     if cur_brt == 0 and cur_ext == 0:
    #         return False
    #
    #     if len(obj_to_add.los_velocity) <= 3:
    #         cur_lineprofile = np.zeros(shape=len(self.vel_grid), dtype=float)
    #         idx = (np.abs(self.vel_grid - obj_to_add.sys_velocity)).argmin()
    #         cur_lineprofile[idx] = 1.
    #     else:
    #         ip = interp1d(obj_to_add.los_velocity, obj_to_add.line_profile, fill_value=0.,
    #                       assume_sorted=True, bounds_error=False, copy=False)
    #         cur_lineprofile = ip(self.vel_grid)
    #         cur_lineprofile[cur_lineprofile < self.vel_contrib_eps] = 0
    #     dx = (len(obj_to_add._cartesian_x_grid) - 1) / 2
    #     dy = (len(obj_to_add._cartesian_y_grid) - 1) / 2
    #     self.content.add_row([self.last_id, x - dx + obj_to_add.xc, y - dy + obj_to_add.yc, cur_brt, cur_ext,
    #                           cur_lineprofile, turb_velocity, spectrum_id, zorder])  # , obj_to_add])
    #     return True

    def add_nebula(self, obj_to_add, obj_id=0, zorder=0):
        """
        Method to add the particular nebula to the ISM object and to the output multi-extensions fits file
        """
        if type(obj_to_add) not in [Nebula, Bubble, Filament, DIG, Cloud]:
            log.warning('Skip nebula of wrong type ({0})'.format(type(obj_to_add)))
            return

        if (obj_to_add.max_brightness <= 0) and (obj_to_add.max_extinction <= 0):
            log.warning('Skip nebula with zero extinction and brightness')
            return

        if obj_to_add.max_brightness > 0:
            brt = obj_to_add.brightness_skyplane
        else:
            brt = obj_to_add.extinction_skyplane

        self.content.append(fits.ImageHDU(brt, name="Comp_{0}_Brightness".format(obj_id)))
        self.content[-1].header['Dark'] = (obj_to_add.max_brightness <= 0)
        self.content[-1].header['X0'] = obj_to_add.x0
        self.content[-1].header['Y0'] = obj_to_add.y0
        self.content[-1].header['Zorder'] = zorder

        # # self._cur_obj_to_add = obj_to_add
        # ny = len(obj_to_add._cartesian_y_grid)
        # nx = len(obj_to_add._cartesian_x_grid)
        # npix = nx * ny
        # xx, yy = np.meshgrid(np.arange(nx), np.arange(ny))
        # params = zip(xx.ravel(), yy.ravel(),
        #              [turb_velocity] * npix, [zorder] * npix, [spectrum_id] * npix)

        # for p in params:
        #     self._add_pix_to_tab(p)
        # with Pool() as p:
        #     r = p.map(self._add_pix_to_tab, params)

        #     _ = list(tqdm.tqdm(p.imap(self._add_pix_to_tab, params), total=npix))

        # for xy in zip(xx.ravel(), yy.ravel()):
        #     if obj_to_add.max_brightness > 0:
        #         cur_brt = obj_to_add.brightness_skyplane[xy[0], xy[1]]
        #         if cur_brt < self.brt_eps:
        #             cur_brt = 0
        #     else:
        #         cur_brt = 0
        #     if obj_to_add.max_extinction > 0:
        #         cur_ext = obj_to_add.extinction[xy[0], xy[1]]
        #         if cur_ext < self.ext_eps:
        #             cur_ext = 0
        #     else:
        #         cur_ext = 0
        #
        #     if cur_brt == 0 and cur_ext == 0:
        #         continue
        #
        #     if len(obj_to_add.los_velocity) <= 3:
        #         cur_lineprofile = np.zeros(shape=len(self.vel_grid), dtype=float)
        #         idx = (np.abs(self.vel_grid - obj_to_add.sys_velocity)).argmin()
        #         cur_lineprofile[idx] = 1.
        #     else:
        #         ip = interp1d(obj_to_add.los_velocity, obj_to_add.line_profile, fill_value=0.,
        #                       assume_sorted=True, bounds_error=False, copy=False)
        #         cur_lineprofile = ip(self.vel_grid)
        #         cur_lineprofile[cur_lineprofile < self.vel_contrib_eps] = 0
        #
        #     self.content.add_row([self.last_id, xy[0], xy[1], cur_brt, cur_ext, cur_lineprofile,
        #                           turb_velocity, spectrum_id, zorder])  # , obj_to_add])
        self.last_id += 1
        return self.content

    def generate(self, all_objects):
        """
        Generate all the Nebulae from the input list

        Args:
            all_objects: list -- contains a dictionary describing the nebula to add:
                example:
                all_objects = [{type: "Bubble, Filament, DIG, ....",
                                sys_velocity: 0 * u.km/u.s,
                                expansion_velocity: 30 * u.km/u.s,
                                turbulent_sigma: 10 * u.km/u.s,
                                radius: 5 * u.pc,
                                max_brightness: 1e-13 * u.erg / u.cm**2 / u.s,
                                RA: "08h12m13s",
                                DEC: "-20d14m13s",
                                'perturb_order': 8, # max. order of spherical harmonics used to generate inhomogeneities
                                'perturb_amplitude': 0.1, # relative max. amplitude of inhomogeneities,
                                'spectrum_id': 4,  # id of the pregenerated Cloudy spectrum
                                'run_cloudy': False  # True if Cloudy modelling should be performed instead
                                }]
        """
        if type(all_objects) not in [list, tuple]:
            log.warning('Cannot generate nebulae as the input is not a list or tuple')
            return
        all_objects = [cobj for cobj in all_objects if cobj.get('type') in ['Nebula', 'Bubble',
                                                                            'Filament', 'DIG', 'Cloud']]
        n_objects = len(all_objects)
        log.info("Start generating {} nebulae".format(n_objects))

        obj_id = self.content[0].header['Nobj']
        for ind_obj, cur_obj in enumerate(all_objects):
            # Setup default parameters for missing keywords
            for k, v in zip(['max_brightness', 'max_extinction', 'thickness',
                             'expansion_velocity', 'sys_velocity',
                             'turbulent_sigma', 'perturb_order',
                             'perturb_amplitude', 'radius'],
                            [0, 0, 1., 0, self.sys_velocity, self.turbulent_sigma, 0, 0.1, 0]):
                set_default_dict_values(cur_obj, k, v)
            for k in ['max_brightness', 'max_extinction', 'radius']:
                if cur_obj[k] < 0:
                    cur_obj[k] = 0

            if (cur_obj['max_brightness'] == 0) and (cur_obj['max_extinction'] == 0):
                log.warning("Wrong set of parameters for the nebula #{0}: skip this one".format(ind_obj))
                continue
            cur_obj['run_cloudy'] = False
            if 'spectrum_id' not in cur_obj:
                if not cur_obj['run_cloudy']:
                    log.warning("Spectrum_id is not set for the nebula #{0}: use spec_id=0".format(ind_obj))
                cur_obj['spectrum_id'] = 0

            if cur_obj['type'] == 'DIG':
                if not cur_obj.get('max_brightness'):
                    log.warning("Wrong set of parameters for the nebula #{0}: skip this one".format(ind_obj))
                    continue
                generated_object = DIG(max_brightness=cur_obj.get('max_brightness'),
                                       turbulent_sigma=cur_obj['turbulent_sigma'],
                                       sys_velocity=cur_obj['sys_velocity'],
                                       vel_gradient=0,
                                       spectrum_id=cur_obj['spectrum_id'],
                                       pxscale=self.pxscale,
                                       )
            else:
                # ==== Check input parameters and do necessary conversions
                if not ((cur_obj.get('RA') and cur_obj.get('DEC')) or
                        (cur_obj.get('X') and cur_obj.get('Y'))):
                    log.warning("Wrong set of parameters for the nebula #{0}: skip this one".format(ind_obj))
                    continue
                if (cur_obj['type'] in ["Bubble", "Cloud"]) and (cur_obj['radius'] == 0):
                    log.warning("Wrong set of parameters for the nebula #{0}: skip this one".format(ind_obj))
                    continue
                if cur_obj['type'] == 'Filament' and not (('length' in cur_obj) and ('PA' in cur_obj)):
                    log.warning("Wrong set of parameters for the nebula #{0}: skip this one".format(ind_obj))
                    continue

                if not (cur_obj.get('X') and cur_obj.get('Y')):
                    radec = SkyCoord(ra=cur_obj.get('RA'), dec=cur_obj.get('DEC'))
                    x, y = self.wcs.all_world2pix(radec)
                else:
                    x, y = [cur_obj.get('X'), cur_obj.get('Y')]

                if (cur_obj['thickness'] <= 0) or (cur_obj['thickness'] > 1):
                    log.warning("Wrong value of thickness of the nebula #{0}: set it to 1.".format(ind_obj))
                    cur_obj['thickness'] = 1.

                if cur_obj['type'] == "Bubble" and cur_obj.get('expansion_velocity') <= 0:
                    log.warning("Contracting bubbles are not supported (nebula #{0})."
                                " Use non-expanding cloud instead".format(ind_obj))
                    cur_obj['type'] = "Cloud"

                if cur_obj['type'] in ["Bubble", "Cloud"]:
                    if cur_obj['perturb_order'] < 0:
                        cur_obj['perturb_order'] = 0

                elif cur_obj['type'] == 'Filament' and ('width' not in cur_obj):
                    cur_obj['width'] = 0.1 * u.pc

                # ==== Start calculations of different nebulae
                if cur_obj['type'] == "Bubble":
                    generated_object = Bubble(xc=x, yc=y,
                                              max_brightness=cur_obj.get('max_brightness'),
                                              max_extinction=cur_obj.get('max_extinction'),
                                              spectral_axis=self.vel_grid,
                                              expansion_velocity=cur_obj.get('expansion_velocity'),
                                              thickness=cur_obj['thickness'],
                                              radius=cur_obj['radius'], pxscale=self.pxscale,
                                              harm_maxdegree=cur_obj['perturb_order'],
                                              harm_amplitude=cur_obj['perturb_amplitude'],
                                              turbulent_sigma=cur_obj['turbulent_sigma'],
                                              sys_velocity=cur_obj['sys_velocity'],
                                              spectrum_id=cur_obj['spectrum_id']
                                              )
                elif cur_obj['type'] == "Cloud":
                    generated_object = Cloud(xc=x, yc=y,
                                             max_brightness=cur_obj.get('max_brightness'),
                                             max_extinction=cur_obj.get('max_extinction'),
                                             thickness=cur_obj['thickness'],
                                             radius=cur_obj['radius'], pxscale=self.pxscale,
                                             harm_maxdegree=cur_obj['perturb_order'],
                                             harm_amplitude=cur_obj['perturb_amplitude'],
                                             spectrum_id=cur_obj['spectrum_id'],
                                             turbulent_sigma=cur_obj['turbulent_sigma'],
                                             sys_velocity=cur_obj['sys_velocity'],
                                             )

                elif cur_obj['type'] == "Filament":
                    generated_object = Filament(xc=x, yc=y,
                                                max_brightness=cur_obj.get('max_brightness'),
                                                max_extinction=cur_obj.get('max_extinction'),
                                                width=cur_obj['width'],
                                                length=cur_obj['length'],
                                                PA=cur_obj['PA'],
                                                vel_gradient=0,
                                                spectrum_id=cur_obj['spectrum_id'],
                                                turbulent_sigma=cur_obj['turbulent_sigma'],
                                                sys_velocity=cur_obj['sys_velocity'],
                                                pxscale=self.pxscale,
                                                )
                else:
                    log.warning("Unexpected type of the nebula #{0}: skip this one".format(ind_obj))
                    continue
            # if type(generated_object) is Bubble:
            #     plt.imshow(generated_object.brightness_skyplane)
            #     plt.show()
            # else:
            #     plt.imshow(generated_object.extinction_skyplane)
            #     plt.show()

            self.add_nebula(generated_object, obj_id=obj_id, zorder=cur_obj.get['zorder'])
            obj_id += 1


#
#
#
# if __name__ == '__main__':
#     # velocities = np.linspace(-70, 70, 25) << velunit
#
#     header = fits.Header()
#     header['CDELT1'] = 1/3600
#     header['CDELT2'] = 1 / 3600
#     header['CRVAL1'] = 10.
#     header['CRVAL2'] = -10.
#     header['CRPIX1'] = 1
#     header['CRPIX2'] = 1
#     wcs = WCS(header)
#     ism = ISM(wcs, width=1000, height=1000)
#
#     ism.generate([{"type": 'Bubble', 'expansion_velocity': 30 * u.km/u.s,
#                                 'turbulent_sigma': 10 * u.km/u.s,
#                                 'radius': 5 * u.pc,
#                                 'max_brightness': 1e-13 * u.erg / u.cm**2 / u.s,
#                                 'X': 600,
#                                 'Y': 400, },
#                   {"type": 'Filament',
#                    'max_extinction': 2 * u.mag,
#                    'X': 100,
#                    'Y': 400, 'zorder': 2, 'length': 17 * u.pc, 'width': 3 * u.pc, 'PA': -30 * u.degree }
#                   ])

    # print(ism.content['Brightness'])
    # n = Bubble()
    # print(ism.add_nebula(n))
    # mxdeg = 8
    # bbl = Bubble(max_brightness=1e-15 * fluxunit, thickness=1, turbulent_sigma=0.1 * velunit,
    #              spectral_axis=velocities, expansion_velocity=25 * velunit, harm_maxdegree=mxdeg, harm_amplitude=0.3)
    # brt_2d = bbl.brightness_skyplane
    # # print(np.max(bbl.brightness_skyplane))
    # fig, ax = plt.subplots()
    # plt.imshow(brt_2d, origin='lower')
    # plt.colorbar()
    #
    # # fig, ax = plt.subplots()
    # # plt.imshow(np.argmax(bbl.vel_field,0), origin='lower')
    #
    # fig, ax = plt.subplots()
    # print(bbl.vel_field.shape)
    # plt.plot(velocities, bbl.vel_field[:, 102, 102])
    # plt.plot(velocities, bbl.vel_field[:, 102, 25])
    # # plt.plot(velocities, np.sum(bbl.vel_field(),(0,1)))
    # from astropy.io import fits
    # # hdu = fits.PrimaryHDU(bbl.vel_field)
    # # hdul = [hdu]
    # fits.writeto("/Users/mors/Science/LVM/test.fits", data=bbl._velocity_3d_cartesian, overwrite=True)
    # plt.show()
