# encoding: utf-8
#
# @Author: Oleg Egorov, Enrico Congiu
# @Date: Nov 15, 2021
# @Filename: ism.py
# @License: BSD 3-Clause
# @Copyright: Oleg Egorov, Enrico Congiu

# import scipy.optimize
# import sys
import os.path
from astropy import units as u
from astropy import constants as c
import numpy as np
from astropy.io import fits
# from matplotlib import pyplot as plt
# from scipy.integrate import nquad
# import tqdm
# from multiprocessing import Pool
from scipy.special import sph_harm
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales
from astropy.coordinates import SkyCoord
from astropy.convolution import convolve, kernels
from dataclasses import dataclass
import functools
from scipy.ndimage.interpolation import map_coordinates
from scipy.interpolate import interp1d
# from typing import List
import lvmdatasimulator
from lvmdatasimulator import log
import progressbar
from joblib import Parallel, delayed
import time
from astropy.convolution import convolve_fft, kernels

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


def sphere_brt_in_line(brt_3d, rad_3d, rad_model, flux_model):
    p = interp1d(rad_model, flux_model, fill_value='extrapolate', assume_sorted=True)
    return p(rad_3d) * brt_3d


def interpolate_sphere_to_cartesian(spherical_array, x_grid=None, y_grid=None, z_grid=None,
                                    rad_grid=None, theta_grid=None, phi_grid=None, pxscale=1. * u.pc):
    x, y, z = np.meshgrid(x_grid, y_grid, z_grid, indexing='ij')
    phi_c, theta_c, rad_c = xyz_to_sphere(x, y, z, pxscale=pxscale)
    ir = interp1d(rad_grid, np.arange(len(rad_grid)), bounds_error=False)
    ith = interp1d(theta_grid, np.arange(len(theta_grid)))
    iphi = interp1d(phi_grid, np.arange(len(phi_grid)))
    new_ir = ir(rad_c.ravel())
    new_ith = ith(theta_c.ravel())
    new_iphi = iphi(phi_c.ravel())
    cart_data = map_coordinates(spherical_array, np.vstack([new_ir, new_ith, new_iphi]),
                                order=1, mode='constant', cval=0)
    return cart_data.reshape([len(x_grid), len(y_grid), len(z_grid)]).T


def limit_angle(value, bottom_limit=0, top_limit=np.pi):
    value[value < bottom_limit] += (top_limit - bottom_limit)
    value[value > top_limit] -= (top_limit - bottom_limit)
    return value


def xyz_to_sphere(x, y, z, pxscale=1. * u.pc):
    phi_c = np.arctan2(y, x)
    rad_c = (np.sqrt(x ** 2 + y ** 2 + z ** 2))
    rad_c[rad_c == 0 * u.pc] = 1e-3 * pxscale
    theta_c = (np.arccos(z / rad_c))
    phi_c = limit_angle(phi_c, 0 * u.radian, 2 * np.pi * u.radian)
    theta_c = limit_angle(theta_c, 0 * u.radian, np.pi * u.radian)
    return phi_c, theta_c, rad_c


def find_model_id(file=lvmdatasimulator.CLOUDY_MODELS,
                  check_id=None, params=lvmdatasimulator.CLOUDY_SPEC_DEFAULTS['id']):
    """
    Checks the input parameters of the pre-computed Cloudy model and return corresponding index in the grid
    """
    with fits.open(file) as hdu:
        if check_id is None:
            if params is None:
                check_id = lvmdatasimulator.CLOUDY_SPEC_DEFAULTS['id']
                log.warning('Default Cloudy model will be used (id = {0})'.format(check_id))
            else:
                for cur_ext in range(len(hdu)):
                    if cur_ext == 0:
                        continue
                    found = False
                    for p in params:
                        if p == 'id':
                            continue
                        precision = 1
                        if p == 'Z':
                            precision = 2
                        if np.round(params[p], precision) != np.round(hdu[cur_ext].header[p], precision):
                            break
                    else:
                        found = True
                    if found:
                        return cur_ext, check_id
                check_id = lvmdatasimulator.CLOUDY_SPEC_DEFAULTS['id']
                log.warning('Input parameters do not correspond any pre-computed Cloudy model.'
                            'Default Cloudy model will be used (id = {0})'.format(check_id))

        extension_index = None
        while extension_index is None:
            extension_index = [cur_ext for cur_ext in range(len(hdu)) if (
                    check_id == hdu[cur_ext].header.get('MODEL_ID'))]
            if len(extension_index) == 0:
                if check_id == lvmdatasimulator.CLOUDY_SPEC_DEFAULTS['id']:
                    log.warning('Model_ID = {0} is not found in the Cloudy models grid. '
                                'Use the first one in the grid instead'.format(check_id))
                    extension_index = 1
                else:
                    log.warning('Model_ID = {0} is not found in the Cloudy models grid. '
                                'Use default ({1}) instead'.format(check_id,
                                                                   lvmdatasimulator.CLOUDY_SPEC_DEFAULTS['id']))
                    check_id = lvmdatasimulator.CLOUDY_SPEC_DEFAULTS['id']
                    extension_index = None
            else:
                extension_index = extension_index[0]
    return extension_index, check_id


def resolve_aperture(cur_wcs, width, height, aperture):
    """
    Construct pixel mask based on the provided WCS, width and height of the field of view and the current aperture

    aperture = {"type": 'circle',  # or 'rect',
                # position of the center; if both "RA, DEC" and "X, Y" are set => "X, Y" will be used
                "X": 100,
                "Y": 300,
                "RA": 128.123443,  # degree
                "DEC": -12.34312,  # degree
                "rad": 12.32 # radius of the circular aperture
                "width": 12.32 # full size of the rectangular aperture
                "height": 12.32 # full size of the rectangular aperture
                "size_unit": 'px'  # or 'degree' or 'arcsec' => unit of the size values (pix is default)
                # if size = 0 => assumed single pixel
                }
    """
    if type(aperture) is not dict or aperture.get('type') not in ['circle', 'rect']:
        log.warning("Unrecognized type of aperture")
        return None
    xx, yy = np.meshgrid(np.arange(width), np.arange(height))
    ap_keys_orig = aperture.keys()
    ap_keys_compar = str.lower(ap_keys_orig)
    pxsize = proj_plane_pixel_scales(cur_wcs)[0] * 3600
    if 'size_unit' not in ap_keys_compar:
        aperture['size_unit'] = 'px'
    elif aperture[ap_keys_orig[ap_keys_compar == 'size_unit']] not in ['px', 'degree', 'arcsec']:
        aperture['size_unit'] = 'px'
    elif aperture[ap_keys_orig[ap_keys_compar == 'size_unit']]:
        pass

    if not ("x" in ap_keys_compar and ("y" in ap_keys_compar)):
        if not ("ra" in ap_keys_compar and ('dec' in ap_keys_compar)):
            log.warning("Incomplete parameters defining aperture")
            return None
        cur_wcs.world_to_pixel(ra="")


@dataclass
class Nebula:
    """
    Base class defining properties of every nebula type.
    By itself it describes the rectangular nebula (e.g. DIG)
    Constructed nebula has 4 dimensions, where 4th derive its appearance in different lines
        (if spectrum_id is None, or if it is dark nebula => only one line)
    """
    xc: int = None  # Center of the region in the field of view, pix
    yc: int = None  # Center of the region in the field of view, pix
    x0: int = 0  # Coordinates of the bottom-left corner in the field of view, pix
    y0: int = 0  # Coordinates of the bottom-left corner in the field of view, pix
    width: int = 101  # full width of cartesian grid, pix (should be odd)
    height: int = 101  # full height of cartesian grid, pix (should be odd)
    pxscale: u.pc = 0.01 * u.pc  # pixel size in pc
    spectrum_id: int = None  # ID of a template Cloudy spectrum for this nebula
    sys_velocity: velunit = 0 * velunit  # Systemic velocity
    turbulent_sigma: velunit = 10 * velunit  # Velocity dispersion due to turbulence; included in calculations of LSF
    max_brightness: fluxunit = 1e-15 * fluxunit
    max_extinction: u.mag = 0 * u.mag
    perturb_scale: int = 0 * u.pc  # Spatial scale of correlated perturbations
    perturb_amplitude: float = 0.1  # Maximal amplitude of perturbations
    _npix_los: int = 1  # full size along line of sight in pixels

    def __post_init__(self):
        if (self.xc is not None) and (self.yc is not None):
            self.x0 = self.xc - np.round((self.width - 1) / 2).astype(int)
            self.y0 = self.yc - np.round((self.height - 1) / 2).astype(int)
        elif (self.x0 is not None) and (self.y0 is not None):
            self.xc = self.x0 + np.round((self.width - 1) / 2).astype(int)
            self.yc = self.y0 + np.round((self.height - 1) / 2).astype(int)
        self._ref_line_id = 0
        self.linerat_constant = True  # True if the ratio of line fluxes shouldn't change across the nebula

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
        brt = np.ones(shape=(self.height, self.width, self._npix_los), dtype=float) / self._npix_los
        if (self.perturb_scale > 0) and (self.perturb_amplitude > 0):
            pertscale = (self.perturb_scale / self.pxscale).value
            perturb = np.random.uniform(-1, 1, (self.height, self.width)) * self.perturb_amplitude / self._npix_los
            xx, yy = np.meshgrid(np.arange(self.width), np.arange(self.height))
            f = np.exp(-2 * (xx ** 2 + yy ** 2) / pertscale)
            perturb = 4 / np.sqrt(np.pi) / pertscale * np.fft.ifft2(np.fft.fft2(perturb) * np.fft.fft2(f)).real
            brt += (perturb[:, :, None] - np.median(perturb))
        return brt

    @functools.cached_property
    def _brightness_4d_cartesian(self):
        """
        Derive the brightness (or density) distribution of the nebula for each emission line in cartesian coordinates
        """
        if self.spectrum_id is None or self.linerat_constant:
            flux_ratios = np.array([1.])
        else:
            with fits.open(lvmdatasimulator.CLOUDY_MODELS) as hdu:
                flux_ratios = hdu[self.spectrum_id].data[1:, 1]
                index_ha = np.flatnonzero(hdu[self.spectrum_id].data[1:, 0] == 6562.81)
                if len(index_ha) == 1:
                    self._ref_line_id = index_ha[0]

        return self._brightness_3d_cartesian[None, :, :, :] * flux_ratios[:, None, None, None]

    @functools.cached_property
    def brightness_skyplane(self):
        if self.max_brightness > 0:
            map2d = np.nansum(self._brightness_3d_cartesian, 2)
            return map2d / np.max(map2d) * self.max_brightness
        else:
            return None

    @functools.cached_property
    def brightness_skyplane_lines(self):
        if self.max_brightness > 0:
            map2d = np.nansum(self._brightness_4d_cartesian, 3)
            return map2d / np.max(map2d[self._ref_line_id, :, :]) * self.max_brightness
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
        self._ref_line_id = 0
        self.linerat_constant = True  # True if the ratio of line fluxes shouldn't change across the nebula

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
    perturb_degree: int = 0  # Degree of perturbations (max. degree of spherical harmonics for cloud)
    linerat_constant: bool = False  # True if the ratio of line fluxes shouldn't change across the nebula
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
        self._ref_line_id = 0

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

        if self.perturb_degree > 0:
            phi_cur = limit_angle(phi + np.random.uniform(0, 2 * np.pi, 1), 0, 2 * np.pi)
            theta_cur = limit_angle(theta + np.random.uniform(0, np.pi, 1), 0, np.pi)
            harm_amplitudes = self.perturb_amplitude * np.random.randn(self.perturb_degree * (self.perturb_degree + 2))

            brt += np.sum(Parallel(n_jobs=lvmdatasimulator.n_process)(delayed(brightness_inhomogeneities_sphere)
                                                                      (harm_amplitudes, ll, phi_cur, theta_cur,
                                                                       rho, med, self.radius, self.thickness)
                                                                      for ll in np.arange(1, self.perturb_degree + 1)),
                          axis=0)
            brt[brt < 0] = 0
        if med > 0:
            brt = brt / np.sum(brt)
        return brt

    @functools.cached_property
    def _brightness_4d_spherical(self):
        """
        Method to calculate brightness of the cloud at given theta, phi and radii for each line

        theta: float -- polar angle [0, np.pi]
        phi: float -- azimuthal angle [0, 2 * np.pi]
        rad: float -- radius [0, self.radius]
        Returns:
            4D cube of brightness in line-theta-phi-rad grid; normalized to the total brightness in Halpha
        """
        s = self._brightness_3d_spherical.shape
        if self.spectrum_id is None or self.linerat_constant:
            return self._brightness_3d_spherical.reshape((1, s[0], s[1], s[2]))
        rho, _, _ = np.meshgrid(self._rad_grid, self._theta_grid, self._phi_grid, indexing='ij')
        with fits.open(lvmdatasimulator.CLOUDY_MODELS) as hdu:
            radius = hdu[self.spectrum_id].data[0, 2:] * (self.thickness * self.radius) + \
                     self.radius * (1 - self.thickness)
            fluxes = hdu[self.spectrum_id].data[1:, 2:]
            index_ha = np.flatnonzero(hdu[self.spectrum_id].data[1:, 0] == 6562.81)
            if len(index_ha) == 1:
                self._ref_line_id = index_ha[0]

            brt = np.array(Parallel(n_jobs=lvmdatasimulator.n_process)(delayed(sphere_brt_in_line)
                                                                       (self._brightness_3d_spherical, rho,
                                                                        radius, flux)
                                                                       for flux in fluxes)).reshape((fluxes.shape[0],
                                                                                                     s[0], s[1], s[2]))
            return brt / np.sum(brt[self._ref_line_id])

    @functools.cached_property
    def _brightness_3d_cartesian(self):
        return interpolate_sphere_to_cartesian(self._brightness_3d_spherical, x_grid=self._cartesian_x_grid,
                                               y_grid=self._cartesian_y_grid, z_grid=self._cartesian_z_grid,
                                               rad_grid=self._rad_grid, theta_grid=self._theta_grid,
                                               phi_grid=self._phi_grid, pxscale=self.pxscale)

    @functools.cached_property
    def _brightness_4d_cartesian(self):
        s = self._brightness_4d_spherical.shape
        return np.array(Parallel(n_jobs=lvmdatasimulator.n_process)(delayed(interpolate_sphere_to_cartesian)
                                                                    (cur_line_array,
                                                                     self._cartesian_x_grid, self._cartesian_y_grid,
                                                                     self._cartesian_z_grid, self._rad_grid,
                                                                     self._theta_grid, self._phi_grid, self.pxscale)
                                                                    for cur_line_array in self._brightness_4d_spherical)
                        ).reshape((s[0], len(self._cartesian_z_grid), len(self._cartesian_y_grid),
                                   len(self._cartesian_x_grid)))


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
        vel_cube[rec] = \
            np.sin(theta[rec]) * \
            np.cos(phi[rec]) * \
            self.expansion_velocity / self._brightness_3d_spherical[rec] * \
            np.median(self._brightness_3d_spherical[self._brightness_3d_spherical > 0])
        return vel_cube

    @functools.cached_property
    def _velocity_3d_cartesian(self) -> velunit:
        return interpolate_sphere_to_cartesian(self._velocity_3d_spherical, x_grid=self._cartesian_x_grid,
                                               y_grid=self._cartesian_y_grid, z_grid=self._cartesian_z_grid,
                                               rad_grid=self._rad_grid, theta_grid=self._theta_grid,
                                               phi_grid=self._phi_grid, pxscale=self.pxscale)

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
    turbulent_sigma: velunit = 10. * velunit  # turbulence vel. disp. to be used for every nebula unless other specified

    # last_id: int = 0
    # ext_eps: u.mag = 0.01 * u.mag
    # brt_eps: fluxunit = 1e-20 * fluxunit
    # vel_contrib_eps: float = 1e-3

    def __post_init__(self):
        self.content = fits.HDUList()
        self.content.append(fits.PrimaryHDU(header=self.wcs.to_header()))
        self.vel_grid = np.linspace(-self.vel_amplitude + self.sys_velocity,
                                    self.vel_amplitude + self.sys_velocity,
                                    np.ceil(self.vel_amplitude / self.vel_resolution).astype(int) * 2 + 1)
        self.pxscale = proj_plane_pixel_scales(self.wcs)[0] * 3600 * self.distance.to(u.pc) / 206265.
        self.content[0].header['width'] = (self.width, "Width of field of view, px")
        self.content[0].header['height'] = (self.height, "Height of field of view, px")
        self.content[0].header['PhysRes'] = (self.pxscale.value, "Physical resolution, pc/arcsec")
        self.content[0].header['Dist'] = (self.distance.value, "Distance, kpc")
        self.content[0].header['Vsys'] = (self.distance.value, "Systemic Velocity, km/s")
        self.content[0].header['VelRes'] = (self.vel_resolution.value, "Velocity resolution, km/s/px")
        self.content[0].header['TurbSig'] = (self.turbulent_sigma.value, "Default turbulent velocity dispersion, km/s")
        self.content[0].header['Nobj'] = (0, "Number of generated nebulae")

    @functools.cached_property
    def vel_resolution(self):
        return (self.spec_resolution / self.npix_line / (10000 * u.Angstrom) * c.c).to(velunit)

    def _add_fits_extension(self, name, value, obj_to_add, zorder=0, cur_wavelength=0, add_fits_kw=None,
                            add_counter=False):
        self.content.append(fits.ImageHDU(value, name=name))
        self.content[-1].header['Nebtype'] = type(obj_to_add).__name__
        self.content[-1].header['Dark'] = (obj_to_add.max_brightness <= 0)
        self.content[-1].header['X0'] = obj_to_add.x0
        self.content[-1].header['Y0'] = obj_to_add.y0
        self.content[-1].header['Zorder'] = zorder
        if type(obj_to_add) in [Bubble, Cloud]:
            self.content[-1].header['Radius'] = obj_to_add.radius.to_value(u.pc)
            self.content[-1].header['PertOrd'] = obj_to_add.perturb_degree
            self.content[-1].header['PertAmp'] = obj_to_add.perturb_amplitude
        if type(obj_to_add) in [Filament]:
            self.content[-1].header['Width'] = obj_to_add.width.to_value(u.pc)
        if obj_to_add.max_brightness <= 0:
            self.content[-1].header['MaxExt'] = obj_to_add.max_extinction.value
        else:
            self.content[-1].header['MaxBrt'] = obj_to_add.max_brightness.value  # .to_value(fluxunit / u.arcsec ** 2)
        if type(obj_to_add) == Bubble:
            self.content[-1].header['Vexp'] = obj_to_add.expansion_velocity.to_value(velunit)
        self.content[-1].header['SysVel'] = obj_to_add.sys_velocity.to_value(velunit)
        self.content[-1].header['TurbVel'] = obj_to_add.turbulent_sigma.to_value(velunit)
        self.content[-1].header['SpecID'] = obj_to_add.spectrum_id
        if cur_wavelength:
            self.content[-1].header['Lambda'] = cur_wavelength
        if add_fits_kw is not None:
            for kw in add_fits_kw:
                self.content[-1].header[kw] = add_fits_kw[kw]
        if add_counter:
            self.content[0].header['Nobj'] += 1

    def add_nebula(self, obj_to_add, obj_id=0, zorder=0, add_fits_kw=None):
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
            brt = obj_to_add.brightness_skyplane.value
            if obj_to_add.spectrum_id is not None and not obj_to_add.linerat_constant:
                brt_4d = obj_to_add.brightness_skyplane_lines.value
            else:
                brt_4d = None
        else:
            brt = obj_to_add.extinction_skyplane.value
            brt_4d = None

        self._add_fits_extension(name="Comp_{0}_Brightness".format(obj_id), value=brt,
                                 obj_to_add=obj_to_add, zorder=zorder, add_fits_kw=add_fits_kw, add_counter=True)
        if obj_to_add.max_brightness > 0:
            if brt_4d is not None:
                with fits.open(lvmdatasimulator.CLOUDY_MODELS) as hdu:
                    wl_list = hdu[obj_to_add.spectrum_id].data[1:, 0]
                for line_ind in range(brt_4d.shape[0]):
                    self._add_fits_extension(name="Comp_{0}_Flux_{1}".format(obj_id, wl_list[line_ind]),
                                             value=brt_4d[line_ind],
                                             obj_to_add=obj_to_add, zorder=zorder, add_fits_kw=add_fits_kw,
                                             cur_wavelength=wl_list[line_ind])
            elif obj_to_add.spectrum_id is not None and obj_to_add.linerat_constant:
                with fits.open(lvmdatasimulator.CLOUDY_MODELS) as hdu:
                    self._add_fits_extension(name="Comp_{0}_FluxRatios".format(obj_id),
                                             value=hdu[obj_to_add.spectrum_id].data[1:, :2].T,
                                             obj_to_add=obj_to_add, zorder=zorder, add_fits_kw=add_fits_kw)

        if type(obj_to_add) == Bubble:
            self._add_fits_extension(name="Comp_{0}_LineProfile".format(obj_id), value=obj_to_add.vel_field.value,
                                     obj_to_add=obj_to_add, zorder=zorder, add_fits_kw=add_fits_kw)
        # self.last_id += 1
        return self.content

    def save_ism(self, filename):
        self.content.writeto(filename, overwrite=True)
        log.info("Generated ISM saved to {0}".format(filename))

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
                                'perturb_degree': 8, # max. order of spherical harmonics to generate inhomogeneities
                                'perturb_amplitude': 0.1, # relative max. amplitude of inhomogeneities,
                                'perturb_scale': 200 * u.pc,  # spatial scale to generate inhomogeneities (DIG only)
                                'cloudy_id': None,  # id of pre-generated Cloudy model
                                'cloudy_params': {'Z': 0.5, 'qH': 49., 'nH': 10, 'Teff': 30000, 'Rin': 0},  #
                                    parameters defining the pre-generated Cloudy model (used if spectrum_id is None)
                                'linerat_constant': False #  if True -> lines ratios doesn't vary across Cloud/Bubble
                                }]
        """
        if type(all_objects) is dict:
            all_objects = [all_objects]
        if type(all_objects) not in [list, tuple]:
            log.warning('Cannot generate nebulae as the input is not a list or tuple')
            return None
        all_objects = [cobj for cobj in all_objects if cobj.get('type') in ['Nebula', 'Bubble',
                                                                            'Filament', 'DIG', 'Cloud']]
        n_objects = len(all_objects)
        log.info("Start generating {} nebulae".format(n_objects))

        obj_id = self.content[0].header['Nobj']
        obj_id_ini = self.content[0].header['Nobj']
        for ind_obj, cur_obj in enumerate(all_objects):
            # Setup default parameters for missing keywords
            for k, v in zip(['max_brightness', 'max_extinction', 'thickness',
                             'expansion_velocity', 'sys_velocity',
                             'turbulent_sigma', 'perturb_degree',
                             'perturb_amplitude', 'perturb_scale', 'radius'],
                            [0, 0, 1., 0, self.sys_velocity, self.turbulent_sigma, 0, 0.1, 0, 0]):
                set_default_dict_values(cur_obj, k, v)
            for k in ['max_brightness', 'max_extinction', 'radius']:
                if cur_obj[k] < 0:
                    cur_obj[k] = 0

            if (cur_obj['max_brightness'] == 0) and (cur_obj['max_extinction'] == 0):
                log.warning("Wrong set of parameters for the nebula #{0}: skip this one".format(ind_obj))
                continue

            if lvmdatasimulator.CLOUDY_MODELS is None or (cur_obj['max_brightness'] <= 0):
                cloudy_model_index = None
                cloudy_model_id = None
            else:
                if cur_obj.get('cloudy_id') is None:
                    if cur_obj.get('cloudy_params') is None or (type(cur_obj.get('cloudy_params')) is not dict):
                        log.warning("Neither of 'cloudy_id' or 'cloudy_params' is set for the nebula #{0}: "
                                    "use default 'cloudy_id={1}'".format(ind_obj,
                                                                         lvmdatasimulator.CLOUDY_SPEC_DEFAULTS['id']))
                        cur_obj['cloudy_id'] = lvmdatasimulator.CLOUDY_SPEC_DEFAULTS['id']
                    else:
                        for p in lvmdatasimulator.CLOUDY_SPEC_DEFAULTS:
                            if p == 'id':
                                continue
                            if cur_obj['cloudy_params'].get(p) is None:
                                cur_obj['cloudy_params'][p] = lvmdatasimulator.CLOUDY_SPEC_DEFAULTS[p]

                cloudy_model_index, cloudy_model_id = find_model_id(file=lvmdatasimulator.CLOUDY_MODELS,
                                                                    check_id=cur_obj.get('cloudy_id'),
                                                                    params=cur_obj.get('cloudy_params'))

            if cur_obj.get('linerat_constant') is None:
                if cur_obj['type'] in ['Bubble', 'Cloud']:
                    cur_obj['linerat_constant'] = False
                else:
                    cur_obj['linerat_constant'] = True

            if cur_obj['type'] == 'DIG':
                if not cur_obj.get('max_brightness'):
                    log.warning("Wrong set of parameters for the nebula #{0}: skip this one".format(ind_obj))
                    continue
                cur_obj['zorder'] = -1
                if cur_obj['perturb_scale'] < 0:
                    cur_obj['perturb_scale'] = 0
                generated_object = DIG(max_brightness=cur_obj.get('max_brightness'),
                                       turbulent_sigma=cur_obj['turbulent_sigma'],
                                       sys_velocity=cur_obj['sys_velocity'],
                                       vel_gradient=0,
                                       spectrum_id=cloudy_model_index,
                                       pxscale=self.pxscale,
                                       perturb_scale=cur_obj['perturb_scale'],
                                       perturb_amplitude=cur_obj['perturb_amplitude'],
                                       width=self.width, height=self.height,
                                       )
            else:
                # ==== Check input parameters and do necessary conversions
                if not cur_obj.get('zorder'):
                    cur_obj['zorder'] = 0
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
                    if cur_obj['perturb_degree'] < 0:
                        cur_obj['perturb_degree'] = 0

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
                                              perturb_degree=cur_obj['perturb_degree'],
                                              perturb_amplitude=cur_obj['perturb_amplitude'],
                                              turbulent_sigma=cur_obj['turbulent_sigma'],
                                              sys_velocity=cur_obj['sys_velocity'],
                                              spectrum_id=cloudy_model_index,
                                              linerat_constant=cur_obj['linerat_constant']
                                              )
                elif cur_obj['type'] == "Cloud":
                    generated_object = Cloud(xc=x, yc=y,
                                             max_brightness=cur_obj.get('max_brightness'),
                                             max_extinction=cur_obj.get('max_extinction'),
                                             thickness=cur_obj['thickness'],
                                             radius=cur_obj['radius'], pxscale=self.pxscale,
                                             perturb_degree=cur_obj['perturb_degree'],
                                             perturb_amplitude=cur_obj['perturb_amplitude'],
                                             spectrum_id=cloudy_model_index,
                                             turbulent_sigma=cur_obj['turbulent_sigma'],
                                             sys_velocity=cur_obj['sys_velocity'],
                                             linerat_constant=cur_obj['linerat_constant']
                                             )

                elif cur_obj['type'] == "Filament":
                    generated_object = Filament(xc=x, yc=y,
                                                max_brightness=cur_obj.get('max_brightness'),
                                                max_extinction=cur_obj.get('max_extinction'),
                                                width=cur_obj['width'],
                                                length=cur_obj['length'],
                                                PA=cur_obj['PA'],
                                                vel_gradient=0,
                                                spectrum_id=cloudy_model_index,
                                                turbulent_sigma=cur_obj['turbulent_sigma'],
                                                sys_velocity=cur_obj['sys_velocity'],
                                                pxscale=self.pxscale,
                                                )
                else:
                    log.warning("Unexpected type of the nebula #{0}: skip this one".format(ind_obj))
                    continue
            if cloudy_model_index is not None:
                if cur_obj['linerat_constant']:
                    lr = "Constant"
                else:
                    lr = "Variable"
                add_fits_kw = {"Model_ID": cloudy_model_id, "LineRat": lr}
            else:
                add_fits_kw = None
            self.add_nebula(generated_object, obj_id=obj_id, zorder=cur_obj.get('zorder'), add_fits_kw=add_fits_kw)
            obj_id += 1
        if (obj_id - obj_id_ini) == 0:
            return None
        else:
            return True

    def load_nebulae(self, file):
        """
        Load previously saved fits-file containing the information about all nebulae.
        Note: Grid should be equal to that used when fits-file was generated!
        """
        if not os.path.isfile(file):
            log.warning("ISM doesn't contain any nebula")
            return None
        else:
            with fits.open(file) as hdu:
                wcs = WCS(hdu[0].header)
                cdelt_file = [cdelt.to(u.degree).value for cdelt in wcs.proj_plane_pixel_scales()]
                cdelt_ism = [cdelt.to(u.degree).value for cdelt in self.wcs.proj_plane_pixel_scales()]
                check = ~np.isclose(cdelt_file, cdelt_ism)
                check = np.append(check, ~np.isclose(wcs.wcs.crval, self.wcs.wcs.crval))
                check = np.append(check, ~np.isclose(hdu[0].header.get("VELRES"), self.vel_resolution.value))
                if any(check):
                    log.warning("Grid of fits file is inconsistent with that defined in ISM")
                    return None
                self.content = fits.HDUList()
                for cur_ind, hh in enumerate(hdu):
                    if cur_ind == 0:
                        self.content.append(fits.PrimaryHDU(header=hh.header))
                    else:
                        self.content.append(fits.ImageHDU(header=hh.header, data=hh.data,
                                                          name=hh.header.get('EXTNAME')))
                return True

    def calc_extinction(self, wavelength=6562.81, x0=0, y0=0, xs=None, ys=None, extinction_name=None, logscale=False):
        """
        Calculate coefficient to reduce flux due to extinction at given wavelength(s)

        Args:
            x0: start x-coordinate in the field of view for calculations
            y0: start x-coordinate in the field of view for calculations
            xs: x-size (in pixels) of the area for calculations (if None => then just pixel x0,y0 is considered; xs=1)
            ys: y-size (in pixels) of the area for calculations (if None => then just pixel x0,y0 is considered; ys=1)
            wavelength: in angstrom, particular wavelength (or wavelengths)
                at which the calculations should be performed
            extinction_name (str): name of the extension for current dark nebula
            logscale: True if the wavelength is np.log(wavelength)
        Returns:
            None (if no any dark nebula at particular location) or np.array of (nlines, ys, xs) shape
        """
        if self.content[0].header['Nobj'] == 0 or (x0 > self.width) or (y0 > self.height):
            return None
        if xs is None:
            x1 = x0
        else:
            x1 = np.clip(x0 + xs - 1, 0, self.width - 1)
        if ys is None:
            y1 = y0
        else:
            y1 = np.clip(y0 + ys - 1, 0, self.height - 1)

        def check_in_region(reg_x0, reg_y0, reg_xs, reg_ys):
            if ((reg_x0 + reg_xs) < x0) or (reg_x0 > x1) or ((reg_y0 + reg_ys) < y0) or (reg_y0 > y1):
                return False
            else:
                return True

        if extinction_name is None:
            all_dark_nebulae = [hdu.header.get('EXTNAME') for hdu in self.content if
                                hdu.header.get('EXTNAME') is not None and
                                ("BRIGHTNESS" in hdu.header.get('EXTNAME')) and
                                hdu.header.get('DARK') and
                                check_in_region(hdu.header.get('X0'), hdu.header.get('Y0'),
                                                hdu.header.get('NAXIS1'), hdu.header.get('NAXIS2'))]
        else:
            if not check_in_region(self.content[extinction_name].header.get('X0'),
                                   self.content[extinction_name].header.get('Y0'),
                                   self.content[extinction_name].header.get('NAXIS1'),
                                   self.content[extinction_name].header.get('NAXIS2')):
                return None
            all_dark_nebulae = [extinction_name]
        if len(all_dark_nebulae) == 0:
            return None
        if type(wavelength) in [float, int]:
            wavelength = np.array([wavelength])
        ext_map = np.ones(shape=(len(wavelength), y1 - y0 + 1, x1 - x0 + 1), dtype=float)
        # NEXT: ADD correct extinction calculations
        # xx, yy = np.meshgrid(np.arange(y1 - y0 + 1), np.arange(x1 - x0 + 1))
        # for xy in zip(xx.ravel(), yy.ravel()):
        #     pass
        return ext_map

    def get_map(self, wavelength=6562.81):
        """
        Method to produce 2D map of all ISM nebulae in the selected line
        Args:
            wavelength (float): exact wavelength (in Angstrom) according to the lines list
        """
        if self.content[0].header['Nobj'] == 0:
            log.warning("ISM doesn't contain any nebula")
            return None
        all_extensions = [hdu.header.get('EXTNAME') for hdu in self.content]
        all_extensions_brt = np.array([extname for extname in all_extensions
                                       if extname is not None and ("BRIGHTNESS" in extname)])

        if all([self.content[cur_ext].header.get("DARK") for cur_ext in all_extensions_brt]):
            log.warning("ISM doesn't contain any emission nebula")
            return None

        all_extensions_brt = all_extensions_brt[
            np.argsort([self.content[cur_ext].header.get('ZORDER') for cur_ext in all_extensions_brt])]

        map_2d = None

        for cur_ext in all_extensions_brt:
            if self.content[cur_ext].header.get("DARK"):
                if map_2d is None:
                    continue
                map_2d = map_2d * self.calc_extinction(wavelength=wavelength, xs=self.width, ys=self.height,
                                                       extinction_name=cur_ext)[0]
                continue
            my_comp = "_".join(cur_ext.split("_")[:2])
            flux_ext = [extname for extname in all_extensions
                        if extname is not None and (my_comp in extname
                                                    and "FLUX_{0}".format(np.round(wavelength, 2)) in extname)]
            if len(flux_ext) == 0:
                fluxrat_ext = [extname for extname in all_extensions
                               if extname is not None and (my_comp in extname and "FLUXRATIOS" in extname)]
                if len(fluxrat_ext) == 0:
                    continue
                fluxrat_ext = fluxrat_ext[0]

                wl_index = np.flatnonzero(np.isclose(self.content[fluxrat_ext].data[0, :], wavelength))
                if len(wl_index) == 0:
                    continue
                add_emission = (self.content[cur_ext].data * self.content[fluxrat_ext].data[1, wl_index[0]])
            else:
                flux_ext = flux_ext[0]
                add_emission = self.content[flux_ext].data

            if map_2d is None:
                map_2d = np.zeros(shape=(self.height, self.width), dtype=float)
            map_2d[self.content[cur_ext].header['Y0']:
                   self.content[cur_ext].header['Y0'] + self.content[cur_ext].header['NAXIS2'],
            self.content[cur_ext].header['X0']:
            self.content[cur_ext].header['X0'] + self.content[cur_ext].header['NAXIS1']] += add_emission
        return map_2d

    def get_spectrum(self, wl_grid=None, aperture_mask=None):
        if aperture_mask is None or (np.sum(aperture_mask) == 0) or (self.content[0].header['Nobj'] == 0):
            return None
        all_extensions = [hdu.header.get('EXTNAME') for hdu in self.content]
        all_extensions_brt = np.array([extname for extname in all_extensions
                                       if extname is not None and ("BRIGHTNESS" in extname)])
        if all([self.content[cur_ext].header.get("DARK") for cur_ext in all_extensions_brt]):
            return None
        all_extensions_brt = all_extensions_brt[np.argsort([self.content[cur_ext].header.get('ZORDER')
                                                            for cur_ext in all_extensions_brt])]

        wl_logscale = np.log(wl_grid.value)
        wl_logscale_highres = np.arange((np.round(wl_logscale[-1] - wl_logscale[0]) * 1e6
                                         ).astype(int)) * 1e-6 + np.round(wl_logscale[0], 6)
        # delta_lr = np.roll(wl_grid.value, -1) - wl_grid.value
        # delta_lr[-1] = delta_lr[-2]
        # wl_highres = np.exp(wl_logscale_highres)
        # delta_hr = np.roll(wl_highres, -1) - wl_highres
        # delta_hr[-1] = delta_hr[-2]
        # p = interp1d(wl_highres, delta_hr, assume_sorted=True)
        # delta_hr = p(wl_grid.value)
        delta_lr = np.roll(wl_logscale, -1) - wl_logscale
        delta_lr[-1] = delta_lr[-2]


        xx, yy = np.meshgrid(np.arange(aperture_mask.shape[1]), np.arange(aperture_mask.shape[0]))
        pix_in_apertures = aperture_mask > 0
        xstart = np.min(xx[pix_in_apertures])
        ystart = np.min(yy[pix_in_apertures])
        xfin = np.max(xx[pix_in_apertures])
        yfin = np.max(yy[pix_in_apertures])
        aperture_mask_sub = aperture_mask[ystart: yfin + 1, xstart: xfin + 1]
        xx_sub, yy_sub = np.meshgrid(np.arange(xfin - xstart + 1), np.arange(yfin - ystart + 1))
        n_apertures = np.max(aperture_mask)
        aperture_centers = np.array([(np.round(np.mean(xx[aperture_mask == (ap_ind+1)])).astype(int),
                                     np.round(np.mean(yy[aperture_mask == (ap_ind+1)])).astype(int))
                                     for ap_ind in range(n_apertures)])
        spectrum = np.zeros(shape=(n_apertures, len(wl_grid)), dtype=float)

        radius = np.round(np.sqrt(np.sum(aperture_mask == n_apertures) / np.pi)).astype(int)
        size = 2 * radius + 1
        if size % 2 == 0:
            size += 1
        kern_array = np.zeros(shape=(1, size, size), dtype=int)
        xx_kern, yy_kern = np.meshgrid(np.arange(size), np.arange(size))
        rec = (xx_kern - radius) ** 2 + (yy_kern - radius) ** 2 <= radius ** 2
        kern_array[0, yy_kern[rec], xx_kern[rec]] = 1
        kern = kernels.CustomKernel(kern_array)

        # tic = time.perf_counter()

        bar = progressbar.ProgressBar(max_value=len(all_extensions_brt)).start()
        for neb_index, cur_ext in enumerate(all_extensions_brt):
            cur_neb_in_mask = np.zeros_like(xx_sub)
            y0 = self.content[cur_ext].header.get("Y0")
            x0 = self.content[cur_ext].header.get("X0")
            nx = self.content[cur_ext].header['NAXIS1']
            ny = self.content[cur_ext].header['NAXIS2']
            cur_neb_in_mask[(xx_sub >= (x0 - xstart)) & (xx_sub <= (x0 + nx - xstart)) &
                            (yy_sub >= (y0 - ystart)) & (yy_sub <= (y0 + ny - ystart))] = True
            cur_neb_in_mask_ap = cur_neb_in_mask * (aperture_mask_sub > 0)
            if not np.sum(cur_neb_in_mask_ap):
                bar.update(neb_index + 1)
                continue
            cur_mask_in_neb = np.zeros(shape=self.content[cur_ext].data.shape, dtype=bool)
            xx_neb, yy_neb = np.meshgrid(np.arange(self.content[cur_ext].data.shape[1]),
                                         np.arange(self.content[cur_ext].data.shape[0]))
            cur_mask_in_neb[(xx_neb >= (xstart - x0)) & (xx_neb <= (xfin - x0)) &
                            (yy_neb >= (ystart - y0)) & (yy_neb <= (yfin - y0))] = True
            xstart_neb = np.min(xx_neb[cur_mask_in_neb])
            ystart_neb = np.min(yy_neb[cur_mask_in_neb])
            xfin_neb = np.max(xx_neb[cur_mask_in_neb])
            yfin_neb = np.max(yy_neb[cur_mask_in_neb])

            selected_apertures = np.flatnonzero((aperture_centers[:, 0] >= x0) & (aperture_centers[:, 0] <= (x0 + nx)) &
                                                (aperture_centers[:, 1] >= y0) & (aperture_centers[:, 1] <= (y0 + ny)))


            if self.content[cur_ext].header.get("DARK"):
                # !!! ADD TREATEMENT OF EXTINCTION !!!
                # for xy in zip(xx[cur_neb_ap].ravel(), yy[cur_neb_ap].ravel()):
                #   self._spectrum[:, xy[1], xy[0]] = self._spectrum[:, xy[1], xy[0]] * \
                #                             self.calc_extinction(wl_logscale_highres, xy[0], xy[1],
                #                                                  extinction_name=cur_ext, logscale=True)
                bar.update(neb_index + 1)
                continue
            my_comp = "_".join(cur_ext.split("_")[:2])

            if self.content[cur_ext].header.get("LINERAT") == 'Variable':
                all_wavelength = np.array([extname.split("_")[-1] for extname in all_extensions
                                           if extname is not None and (my_comp + "_FLUX_" in extname)])
            else:
                all_wavelength = self.content[my_comp + "_FLUXRATIOS"].data[0, :]

            if self.content[cur_ext].header.get("LINERAT") == 'Variable':
                all_fluxes = np.array([self.content[my_comp + "_FLUX_" + wl].data[
                                           cur_mask_in_neb].reshape((yfin_neb - ystart_neb + 1,
                                                                     xfin_neb - xstart_neb + 1))
                                       for wl in all_wavelength])
                all_wavelength = all_wavelength.astype(float)
            else:
                all_fluxes = self.content[cur_ext].data[cur_mask_in_neb].reshape((1, yfin_neb - ystart_neb + 1,
                                                                                  xfin_neb - xstart_neb + 1))

            if my_comp + "_LINEPROFILE" in self.content:
                lsf = self.content[my_comp+"_LINEPROFILE"].data[
                      :, cur_mask_in_neb].reshape((len(self.vel_grid), yfin_neb - ystart_neb + 1,
                                                   xfin_neb - xstart_neb + 1))
            else:
                if my_comp + "_VEL" in self.content:
                    vel = self.content[my_comp + "_VEL"].data[cur_mask_in_neb].reshape((yfin_neb - ystart_neb + 1,
                                                                                        xfin_neb - xstart_neb + 1))
                else:
                    vel = np.zeros(shape=(yfin_neb - ystart_neb + 1, xfin_neb - xstart_neb + 1),
                                   dtype=float) + self.sys_velocity.value
                if my_comp + "_DISP" in self.content:
                    disp = self.content[my_comp + "_DISP"].data[
                        cur_mask_in_neb].reshape((yfin_neb - ystart_neb + 1, xfin_neb - xstart_neb + 1))
                else:
                    disp = np.zeros(shape=(yfin_neb - ystart_neb + 1, xfin_neb - xstart_neb + 1),
                                    dtype=float) + self.turbulent_sigma.value
                lsf = np.exp(-np.power((self.vel_grid.value[:, None, None] - vel[None, :, :]) / disp[
                                                                                                None, :, :], 2.) / 2)
                lsf = lsf / np.sum(lsf)
            data_in_apertures = np.array([convolve_fft(lsf * line_data[None, :, :], kern, normalize_kernel=False)[:,
                                          aperture_centers[selected_apertures, 1] - ystart_neb - y0,
                                          aperture_centers[selected_apertures, 0] - xstart_neb - x0]
                                          for line_data in all_fluxes])

            data_in_apertures = np.moveaxis(data_in_apertures, 2, 0)
            if data_in_apertures.shape[1] > 1:
                prf_index = np.flatnonzero(all_wavelength == 6562.81)
            else:
                prf_index = 0
            flux_norm_in_apertures = data_in_apertures.sum(axis=2)
            line_prf_in_apertures = data_in_apertures[
                                    :, prf_index, :].reshape(
                (data_in_apertures.shape[0], data_in_apertures.shape[2])) / flux_norm_in_apertures[
                                                                            :, prf_index].reshape(
                data_in_apertures.shape[0], 1)

            wl_logscale_lsf = np.log(self.vel_grid.value / 2.9979e5 + 1)
            wl_logscale_lsf_highres = np.arange(np.round((wl_logscale_lsf[-1] - wl_logscale_lsf[0]) * 1e6
                                                         ).astype(int)) * 1e-6 + wl_logscale_lsf[0]
            p = interp1d(wl_logscale_lsf, line_prf_in_apertures, axis=1, assume_sorted=True)
            line_highres_log = p(wl_logscale_lsf_highres)
            line_highres_log = line_highres_log / np.sum(line_highres_log, axis=1)[:, None]
            if flux_norm_in_apertures.shape[1] == 1:
                flux_norm_in_apertures = flux_norm_in_apertures * \
                                    self.content[my_comp + "_FLUXRATIOS"].data[1, None, :]

            wl_indexes = np.round((np.log(all_wavelength) - wl_logscale_highres[0]) * 1e6).astype(int)
            rec = (wl_indexes > 0) & (wl_indexes < len(wl_logscale_highres))
            spectrum_highres_log = np.zeros(shape=(len(selected_apertures), len(wl_logscale_highres)), dtype=float)
            win = (len(wl_logscale_lsf_highres) - 1) // 2
            for ind, r in enumerate(rec):
                if r:
                    spectrum_highres_log[:, wl_indexes[ind] - win: wl_indexes[ind] + win + 1] += \
                        line_highres_log[:, :] * flux_norm_in_apertures[:, ind].reshape((len(selected_apertures), 1))
            p = interp1d(wl_logscale_highres, spectrum_highres_log, axis=1, assume_sorted=True)
            spectrum[selected_apertures, :] += (p(wl_logscale) * delta_lr * 1e6)
            # print(np.sum(flux_norm_in_apertures, axis=1), np.sum((p(wl_logscale) * delta_lr * 1e6), axis=1))
            bar.update(neb_index + 1)
        bar.finish()
        # toc = time.perf_counter()
        # log.info("ISM Spectrum created in {:.0f}s".format(toc - tic))

        return spectrum * fluxunit

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