# encoding: utf-8
#
# @Author: Oleg Egorov, Enrico Congiu
# @Date: Nov 15, 2021
# @Filename: ism.py
# @License: BSD 3-Clause
# @Copyright: Oleg Egorov, Enrico Congiu

import os.path
from astropy import units as u
from astropy import constants as c
from lvmdatasimulator.exceptions import ModelsError
import numpy as np
from astropy.io import fits, ascii
from astropy.table import Table
from scipy.special import sph_harm
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales
from astropy.coordinates import SkyCoord
from astropy.modeling.models import Sersic2D
from dataclasses import dataclass
import sys
if (sys.version_info[0]+sys.version_info[1]/10.) < 3.8:
    from backports.cached_property import cached_property
else:
    from functools import cached_property
from scipy.ndimage.interpolation import map_coordinates
from scipy.interpolate import interp1d, RectBivariateSpline
import lvmdatasimulator
from lvmdatasimulator import log
import progressbar
from joblib import Parallel, delayed
from astropy.convolution import convolve_fft, kernels
from astropy.units import UnitConversionError
from lvmdatasimulator.utils import calc_circular_mask, convolve_array, set_default_dict_values, \
    ism_extinction, check_overlap, assign_units

import warnings

fluxunit = u.erg / (u.cm ** 2 * u.s * u.arcsec ** 2)
velunit = u.km / u.s


def brightness_inhomogeneities_sphere(harm_amplitudes, ll, phi_cur, theta_cur, rho, med, radius, thickness):
    """
    Auxiliary function producing the inhomogeneities on the brightness distribution for the Cloud of Bubble objects
    using the spherical harmonics.
    """
    brt = theta_cur * 0
    for m in np.arange(-ll, ll + 1):
        brt += (harm_amplitudes[m + ll * (ll + 1) - 1] * sph_harm(m, ll, phi_cur, theta_cur).real * med *
                (1 - np.sqrt(abs(rho.value ** 2 / radius.value ** 2 - (1 - thickness / 2) ** 2))))
    return brt


def sphere_brt_in_line(brt_3d, rad_3d, rad_model, flux_model):
    """
    Auxiliary function computing the brightness of the Cloud or Bubble at given radii and in given line
    according to the Cloudy models
    """
    p = interp1d(rad_model, flux_model, fill_value='extrapolate', assume_sorted=True)
    return p(rad_3d) * brt_3d


def interpolate_sphere_to_cartesian(spherical_array, x_grid=None, y_grid=None, z_grid=None,
                                    rad_grid=None, theta_grid=None, phi_grid=None, pxscale=1. * u.pc):
    """
    Auxiliary function to project the brightness or velocities from the spherical to cartesian coordinates
    """
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
    """
    Auxiliary function to limit the angle values to the range of [0, pi]
    """
    value[value < bottom_limit] += (top_limit - bottom_limit)
    value[value > top_limit] -= (top_limit - bottom_limit)
    return value


def xyz_to_sphere(x, y, z, pxscale=1. * u.pc):
    """
    Auxiliary function to map the coordinates from cartesian to spherical system
    """
    phi_c = np.arctan2(y, x)
    rad_c = (np.sqrt(x ** 2 + y ** 2 + z ** 2))
    rad_c[rad_c == 0 * u.pc] = 1e-3 * pxscale
    theta_c = (np.arccos(z / rad_c))
    phi_c = limit_angle(phi_c, 0 * u.radian, 2 * np.pi * u.radian)
    theta_c = limit_angle(theta_c, 0 * u.radian, np.pi * u.radian)
    return phi_c, theta_c, rad_c


def find_model_id(file=lvmdatasimulator.CLOUDY_MODELS,
                  check_id=None, params=lvmdatasimulator.CLOUDY_SPEC_DEFAULTS['id'],
                  defaults=lvmdatasimulator.CLOUDY_SPEC_DEFAULTS['id']):
    """
    Checks the input parameters of the pre-computed Cloudy model and return corresponding index in the grid
    """

    if file is None:
        return None, None

    with fits.open(file) as hdu:
        if check_id is None:
            if params is None:
                check_id = defaults
                log.warning(f'Default Cloudy model will be used (id = {check_id})')
            else:
                summary_table = Table(hdu['Summary'].data)
                indexes = np.arange(len(summary_table)).astype(int)
                rec_table = np.ones(shape=len(summary_table), dtype=bool)

                def closest(rec, prop, val):
                    unique_col = np.unique(summary_table[prop][rec])
                    if isinstance(val, str):
                        res = unique_col[unique_col == val]
                        if len(res) == 0:
                            return ""
                        return res
                    else:
                        return unique_col[np.argsort(np.abs(unique_col - val))[0]]

                for p in params:
                    if p not in summary_table.colnames or params[p] is None or \
                            ((isinstance(params[p], float) or isinstance(params[p], int)) and ~np.isfinite(params[p])):
                        continue
                    rec_table = rec_table & (summary_table[p] == closest(indexes, p, params[p]))
                    indexes = np.flatnonzero(rec_table)
                    if len(indexes) == 0:
                        break
                if len(indexes) == 0 or len(indexes) == len(summary_table):
                    check_id = defaults
                    log.warning('Input parameters do not correspond to any pre-computed model.'
                                'Default model will be used (id = {0})'.format(check_id))
                elif len(indexes) == 1:
                    check_id = summary_table['Model_ID'][indexes[0]]
                    for p in params:
                        if p not in summary_table.colnames or params[p] is None or \
                                ((isinstance(params[p], float) or
                                  isinstance(params[p], int)) and ~np.isfinite(params[p])):
                            continue
                        if params[p] != summary_table[p][indexes[0]]:
                            log.warning(f'Use the closest pre-computed model with id = {check_id}')
                            break
                else:
                    check_id = summary_table['Model_ID'][indexes[0]]
                    log.warning(f'Select one of the closest pre-computed model with id = {check_id}')
                #
                # for cur_ext in range(len(hdu)):
                #     if cur_ext == 0:
                #         continue
                #     found = False
                #     for p in params:
                #         if p == 'id':
                #             continue
                #         precision = 1
                #         if p == 'Z':
                #             precision = 2
                #         if np.round(params[p], precision) != np.round(hdu[cur_ext].header[p], precision):
                #             break
                #     else:
                #         found = True
                #     if found:
                #         return cur_ext, check_id
                # check_id = lvmdatasimulator.CLOUDY_SPEC_DEFAULTS['id']
                # log.warning('Input parameters do not correspond to any pre-computed Cloudy model.'
                #             'Default Cloudy model will be used (id = {0})'.format(check_id))

        extension_index = None
        while extension_index is None:
            extension_index = [cur_ext for cur_ext in range(len(hdu)) if (
                    check_id == hdu[cur_ext].header.get('MODEL_ID'))]
            if len(extension_index) == 0:
                if check_id == defaults:
                    log.warning('Model_ID = {0} is not found in the models grid. '
                                'Use the first one in the grid instead'.format(check_id))
                    extension_index = 1
                else:
                    log.warning('Model_ID = {0} is not found in the models grid. '
                                'Use default ({1}) instead'.format(check_id,
                                                                   defaults))
                    check_id = defaults
                    extension_index = None
            else:
                extension_index = extension_index[0]
    return extension_index, check_id


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
    pix_width: int = None  # full width of cartesian grid, pix (should be odd)
    pix_height: int = None  # full height of cartesian grid, pix (should be odd)
    width: u.pc = 0 * u.pc  # width of the nebula in pc (not used if pix_width is set up)
    height: u.pc = 0 * u.pc  # height of the nebula in pc (not used if pix_height is set up)
    pxscale: u.pc = 0.01 * u.pc  # pixel size in pc
    spectrum_id: int = None  # ID of a template emission spectrum for this nebula
    spectrum_type: str = None  # type of model spectrum (either mappings or cloudy)
    n_brightest_lines: int = None  # limit the number of the lines to the first N brightest
    sys_velocity: velunit = 0 * velunit  # Systemic velocity
    turbulent_sigma: velunit = 10 * velunit  # Velocity dispersion due to turbulence; included in calculations of LSF
    max_brightness: fluxunit = 1e-15 * fluxunit
    max_extinction: u.mag = 0 * u.mag
    perturb_scale: int = 0 * u.pc  # Spatial scale of correlated perturbations
    perturb_amplitude: float = 0.1  # Maximal amplitude of perturbations
    _npix_los: int = 1  # full size along line of sight in pixels
    nchunks: int = -1  # number of chuncks to use for the convolution. If negative, select automatically
    vel_gradient: (velunit / u.pc) = 0  # velocity gradient along the nebula
    vel_pa: u.degree = 0  # Position angle of the kinematical axis (for the velocity gradient or rotation velocity)

    def __post_init__(self):
        self._assign_all_units()
        self._assign_position_params()
        self._ref_line_id = 0
        self.linerat_constant = True  # True if the ratio of line fluxes shouldn't change across the nebula
        if self.spectrum_type not in [None, 'mappings', 'cloudy']:
            raise ModelsError('These models are not supported')


    def _assign_all_units(self):
        whole_list_properties = ['pxscale', 'sys_velocity', 'turbulent_sigma', 'max_brightness', 'max_extinction',
                                 'perturb_scale', 'radius', 'PA', 'length', 'width', 'vel_gradient', 'r_eff',
                                 'vel_rot', 'expansion_velocity', 'spectral_axis', 'vel_pa']
        whole_list_units = [u.pc, velunit, velunit, fluxunit, u.mag, u.pc, u.pc, u.degree, u.pc, u.pc,
                            (velunit / u.pc), u.kpc, velunit, velunit, velunit, u.degree]
        cur_list_properties = []
        cur_list_units = []
        for prp, unit in zip(whole_list_properties, whole_list_units):
            if hasattr(self, prp):
                cur_list_properties.append(prp)
                cur_list_units.append(unit)
        assign_units(self, cur_list_properties, cur_list_units)

    def _assign_position_params(self, conversion_type='rect'):
        if conversion_type == 'rect':
            for v in ['height', 'width']:
                if self.__getattribute__(f'pix_{v}') is None:
                    val = np.round((self.__getattribute__(v) / self.pxscale).value / 2.).astype(int) * 2 + 1
                else:
                    val = np.round(self.__getattribute__(f'pix_{v}') / 2.).astype(int) * 2 + 1
                setattr(self, f'pix_{v}', val)
        elif conversion_type == 'ellipse':
            self.pix_width = (np.round(np.abs(self.radius / self.pxscale * np.sin(self.PA)) +
                              np.abs(self.radius / self.pxscale *
                                     self.ax_ratio * np.cos(self.PA))).astype(int) * 2 + 1).value
            self.pix_height = (np.round(np.abs(self.radius / self.pxscale * np.cos(self.PA)) +
                               np.abs(self.radius / self.pxscale *
                                      self.ax_ratio * np.sin(self.PA))).astype(int) * 2 + 1).value
        elif conversion_type == 'galaxy':
            self.pix_width = (np.round(np.abs(self.r_max * np.sin(self.PA)) +
                                       np.abs(self.r_max * self.ax_ratio * np.cos(self.PA))).astype(int) * 2 + 1).value
            self.pix_height = (np.round(np.abs(self.r_max * np.cos(self.PA)) +
                                        np.abs(self.r_max * self.ax_ratio * np.sin(self.PA))).astype(int) * 2 + 1).value
        elif conversion_type == 'cylinder':
            self.pix_width = (np.ceil((self.length * np.abs(np.sin(self.PA)) +
                                       self.width * np.abs(np.cos(self.PA))) / self.pxscale / 2.
                                      ).astype(int) * 2 + 1).value
            self.pix_height = (np.ceil((self.length * np.abs(np.cos(self.PA)) +
                                       self.width * np.abs(np.sin(self.PA))) / self.pxscale / 2.
                                       ).astype(int) * 2 + 1).value

        if (self.xc is not None) and (self.yc is not None):
            self.x0 = self.xc - np.round((self.pix_width - 1) / 2).astype(int)
            self.y0 = self.yc - np.round((self.pix_height - 1) / 2).astype(int)
        elif (self.x0 is not None) and (self.y0 is not None):
            self.xc = self.x0 + np.round((self.pix_width - 1) / 2).astype(int)
            self.yc = self.y0 + np.round((self.pix_height - 1) / 2).astype(int)

    @cached_property
    def _cartesian_x_grid(self):
        return np.arange(self.pix_width) * self.pxscale

    @cached_property
    def _cartesian_y_grid(self):
        return np.arange(self.pix_height) * self.pxscale

    @cached_property
    def _cartesian_z_grid(self):
        return np.arange(self._npix_los) * self.pxscale

    @cached_property
    def _max_density(self):
        return self.max_extinction * (1.8e21 / (u.cm ** 2 * u.mag))

    @cached_property
    def _brightness_3d_cartesian(self):
        """
        Method to obtain the brightness (or density) distribution of the nebula in cartesian coordinates
        """
        brt = np.ones(shape=(self.pix_height, self.pix_width, self._npix_los), dtype=float) / self._npix_los
        if (self.perturb_scale > 0) and (self.perturb_amplitude > 0):
            pertscale = (self.perturb_scale / self.pxscale).value
            perturb = np.random.uniform(-1, 1, (self.pix_height, self.pix_width)
                                        ) * self.perturb_amplitude / self._npix_los
            xx, yy = np.meshgrid(np.arange(self.pix_width), np.arange(self.pix_height))
            f = np.exp(-2 * (xx ** 2 + yy ** 2) / pertscale)
            perturb = 4 / np.sqrt(np.pi) / pertscale * np.fft.ifft2(np.fft.fft2(perturb) * np.fft.fft2(f)).real
            brt += (perturb[:, :, None] - np.median(perturb))
        return brt

    @cached_property
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
                if self.n_brightest_lines is not None and \
                        (self.n_brightest_lines > 0) and (self.n_brightest_lines < len(flux_ratios)):
                    indexes_sorted = np.argsort(flux_ratios)[::-1]
                    flux_ratios = flux_ratios[indexes_sorted[: self.n_brightest_lines]]
                    index_ha = np.flatnonzero(hdu[self.spectrum_id].data[1:, 0][indexes_sorted] == 6562.81)
                if len(index_ha) == 1:
                    self._ref_line_id = index_ha[0]
        return self._brightness_3d_cartesian[None, :, :, :] * flux_ratios[:, None, None, None]

    @cached_property
    def brightness_skyplane(self):
        """
        Project the 3D nebula onto sky plane (for emission or continuum sources)
        """
        if self.max_brightness > 0:
            norm_max = self.max_brightness
        else:
            norm_max = 1
        map2d = np.nansum(self._brightness_3d_cartesian, 2)
        return map2d / np.nanmax(map2d) * norm_max

    @cached_property
    def brightness_skyplane_lines(self):
        """
        Project the 3D emission nebula line onto sky plane (return images in each emission line)
        """
        if self.max_brightness > 0:
            map2d = np.nansum(self._brightness_4d_cartesian, 3)
            return map2d / np.nanmax(map2d[self._ref_line_id, :, :]) * self.max_brightness
        else:
            return None

    @cached_property
    def extinction_skyplane(self):
        """
        Project the 3D nebula onto sky plane (for dark clouds)
        """
        if self.max_extinction > 0:
            map2d = np.nansum(self._brightness_3d_cartesian, 2)
            return map2d / np.nanmax(map2d) * self._max_density / (1.8e21 / (u.cm ** 2 * u.mag))
        else:
            return None

    @cached_property
    def vel_field(self):
        return self._get_2d_velocity()
        # if vel_field is None:
        #     return np.atleast_1d(self.sys_velocity)
        # else:
        #     return vel_field + self.sys_velocity

    def _get_2d_velocity(self):
        if hasattr(self, 'vel_gradient') and (self.vel_gradient is not None) and (self.vel_gradient != 0):
            xx, yy = np.meshgrid(np.arange(self.pix_width), np.arange(self.pix_height))
            vel_field = (- (xx - (self.pix_width - 1) / 2) * np.sin(self.vel_pa) +
                         (yy - (self.pix_height - 1) / 2) * np.cos(self.vel_pa)) * self.pxscale * self.vel_gradient
            return vel_field
        else:
            return None

    # @cached_property
    # def line_profile(self):
    #     lprf = np.zeros(shape=len(self.los_velocity), dtype=float)
    #     lprf[np.floor(len(lprf) / 2.).astype(int)] = 1.
    #     return lprf


@dataclass
class Rectangle(Nebula):
    """
    Class defining a simple rectangular component.
    This is equal to Nebula, but no perturbations and turbulence by default
    """
    perturb_amplitude: float = 0.0  # Maximal amplitude of perturbations
    turbulent_sigma: velunit = 0 * velunit  # Velocity dispersion due to turbulence; included in calculations of LSF

    def __post_init__(self):
        self._assign_all_units()
        self._assign_position_params()
        self._ref_line_id = 0
        self.linerat_constant = True  # True if the ratio of line fluxes shouldn't change across the nebula


@dataclass
class Ellipse(Nebula):
    """
    Class defining a simple elliptical component.
    No perturbations and turbulence by default
    """
    perturb_amplitude: float = 0.0  # Maximal amplitude of perturbations
    turbulent_sigma: velunit = 0 * velunit  # Velocity dispersion due to turbulence; included in calculations of LSF
    radius: u.pc = 1.0 * u.pc  # Radius along the major axis of the ellipse (or radius of the circle)
    PA: u.degree = 90 * u.degree  # position angle of the major axis
    ax_ratio: float = 1.  # ratio of minor/major axes

    def __post_init__(self):
        self._assign_all_units()
        self._npix_los = 1
        self._assign_position_params(conversion_type='ellipse')
        self._ref_line_id = 0
        self.linerat_constant = True  # True if the ratio of line fluxes shouldn't change across the nebula

    @cached_property
    def _brightness_3d_cartesian(self):
        """
        Method to obtain the brightness (or density) distribution of the nebula in cartesian coordinates
        """
        xx, yy = np.meshgrid(np.arange(self.pix_width), np.arange(self.pix_height))
        brt = np.ones(shape=(self.pix_height, self.pix_width), dtype=np.float32)
        angle = (self.PA + 90 * u.degree).to(u.radian).value
        xct = (xx - (self.pix_width - 1) / 2) * np.cos(angle) + \
              (yy - (self.pix_height - 1) / 2) * np.sin(angle)
        yct = (xx - (self.pix_width - 1) / 2) * np.sin(angle) - \
              (yy - (self.pix_height - 1) / 2) * np.cos(angle)
        rmaj = (self.radius.to(u.pc) / self.pxscale.to(u.pc)).value
        rmin = (self.radius.to(u.pc) / self.pxscale.to(u.pc)).value * self.ax_ratio
        rec = (xct ** 2 / rmaj ** 2) + (yct ** 2 / rmin ** 2) >= 1
        brt[rec] = 0
        brt = brt.reshape((self.pix_height, self.pix_width, 1))
        return brt


@dataclass
class Circle(Ellipse):
    """
    Class defining a simple circular component.
    """
    def __post_init__(self):
        self._assign_all_units()
        self.ax_ratio = 1.
        self._npix_los = 1
        self._assign_position_params(conversion_type='ellipse')
        self._ref_line_id = 0
        self.linerat_constant = True  # True if the ratio of line fluxes shouldn't change across the nebula


@dataclass
class Filament(Nebula):
    """
    Class of an isotropic cylindrical shape filament.
    Defined by its position, length, PA, radius, maximal optical depth.
    If it is emission-type filament, then also maximal brightness is required.
    Velocity gradient also can be set up
    """
    PA: u.degree = 90 * u.degree  # position angle of the filament
    length: u.pc = 10 * u.pc  # full length of the filament
    width: u.pc = 0.1 * u.pc  # full width (diameter) of the filament

    def __post_init__(self):
        self._assign_all_units()
        self._assign_position_params(conversion_type='cylinder')
        self._npix_los = 1
        self._ref_line_id = 0
        self.linerat_constant = True  # True if the ratio of line fluxes shouldn't change across the nebula

    @cached_property
    def _brightness_3d_cartesian(self):
        """
        Method to obtain the brightness (or density) distribution of the nebula in cartesian coordinates
        """
        xx, yy = np.meshgrid(np.arange(self.pix_width), np.arange(self.pix_height))
        brt = np.zeros_like(xx, dtype=np.float32)

        xct = (xx - (self.pix_width - 1) / 2) * np.cos(self.PA + 90 * u.degree) + \
              (yy - (self.pix_height - 1) / 2) * np.sin(self.PA + 90 * u.degree)
        yct = (xx - (self.pix_width - 1) / 2) * np.sin(self.PA + 90 * u.degree) - \
              (yy - (self.pix_height - 1) / 2) * np.cos(self.PA + 90 * u.degree)

        rad = ((self.width / self.pxscale).value / 2.)
        len_px = ((self.length / self.pxscale).value / 2.)
        rec = (np.abs(yct) <= rad) & (np.abs(xct) <= len_px)
        brt[rec] = np.sqrt(1. - (yct[rec] / rad) ** 2)
        brt = brt.reshape((self.pix_height, self.pix_width, 1))
        return brt


@dataclass
class _ObsoleteFilament(Nebula):
    """
    Class of an isotropic cylindrical shape filament.
    Defined by its position, length, PA, radius, maximal optical depth
    if it is emission-type filament, then maximal brightness
    NB: this class is obsolete, but might be considered later in case of implementation of varying line ratios
    """
    PA: u.degree = 90 * u.degree  # position angle of the filament
    length: u.pc = 10 * u.pc  # full length of the filament
    width: u.pc = 0.1 * u.pc  # full width (diameter) of the filament
    vel_gradient: (velunit / u.pc) = 0  # velocity gradient along the filament (to be added)
    _theta_bins: int = 50
    _rad_bins: int = 0
    _h_bins: int = 2
    _npix_los: int = 101

    def __post_init__(self):
        self._assign_all_units()
        if self._rad_bins == 0:
            self._rad_bins = np.ceil(self.width.to(u.pc).value / self.pxscale.to(u.pc).value * 5).astype(int)
        if (self.xc is not None) and (self.yc is not None):
            self.x0 = self.xc - np.round((len(self._cartesian_y_grid) - 1) / 2).astype(int)
            self.y0 = self.yc - np.round((len(self._cartesian_z_grid) - 1) / 2).astype(int)
        elif (self.x0 is not None) and (self.y0 is not None):
            self.xc = self.x0 + np.round((len(self._cartesian_y_grid) - 1) / 2).astype(int)
            self.yc = self.y0 + np.round((len(self._cartesian_z_grid) - 1) / 2).astype(int)
        self._ref_line_id = 0
        self.linerat_constant = True  # True if the ratio of line fluxes shouldn't change across the nebula

    @cached_property
    def _theta_grid(self):
        return np.linspace(0, 2 * np.pi, self._theta_bins)

    @cached_property
    def _h_grid(self):
        return np.linspace(0, self.length, self._h_bins)

    @cached_property
    def _rad_grid(self):
        return np.linspace(0, self.width / 2, self._rad_bins)

    @cached_property
    def _cartesian_y_grid(self):
        npix = np.ceil(1.01 * (self.length * np.abs(np.sin(self.PA)) +
                               self.width * np.abs(np.cos(self.PA))) / self.pxscale).astype(int)
        npix_l = npix / 2 - np.ceil(self.length / 2 * np.sin(-self.PA) / self.pxscale).astype(int)
        return (np.linspace(0, npix, npix + 1) - npix_l) * self.pxscale

    @cached_property
    def _cartesian_z_grid(self):
        npix = np.ceil(1.01 * (self.length * np.abs(np.cos(self.PA)) +
                               self.width * np.abs(np.sin(self.PA))) / self.pxscale).astype(int)
        npix_l = npix / 2 - np.ceil(self.length / 2 * np.cos(-self.PA) / self.pxscale).astype(int)
        return (np.linspace(0, npix, npix + 1) - npix_l) * self.pxscale

    @cached_property
    def _cartesian_x_grid(self):
        return np.linspace(-1.01, 1.01, self._npix_los) * self.width / 2

    @cached_property
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

    @cached_property
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
class Galaxy(Nebula):
    """
    Class defining the galaxy object (set up it as Sersic2D profile assuming it has continuum and emission components)
    """
    PA: u.degree = 90 * u.degree  # position angle of the major axis
    ax_ratio: float = 0.7  # ratio of minor/major axes
    r_eff: u.kpc = 1 * u.kpc  # Effective radius in kpc
    rad_lim: float = 3.  # Maximum radius for calculations (in R_eff)
    n: float = 1.  # Sersic index
    vel_rot: velunit = 0 * velunit  # Rotational velocity (not implemented yet)

    def __post_init__(self):
        self._assign_all_units()
        self._npix_los = 1
        self.r_max = self.r_eff.to(u.pc).value / self.pxscale.to(u.pc).value * self.rad_lim
        self._assign_position_params(conversion_type='galaxy')
        self._ref_line_id = 0
        self.linerat_constant = True  # True if the ratio of line fluxes shouldn't change across the nebula

    @cached_property
    def _brightness_3d_cartesian(self):
        """
        Method to obtain the brightness (or density) distribution of the nebula in cartesian coordinates
        """
        xx, yy = np.meshgrid(np.arange(self.pix_width), np.arange(self.pix_height))
        angle = (self.PA + 90 * u.degree).to(u.radian).value
        mod = Sersic2D(amplitude=1, r_eff=(self.r_eff.to(u.pc) / self.pxscale.to(u.pc)).value,
                       n=self.n, x_0=(self.pix_width - 1) / 2, y_0=(self.pix_height - 1) / 2,
                       ellip=1 - self.ax_ratio, theta=angle)
        brt = mod(xx, yy)

        xct = (xx - (self.pix_width - 1) / 2) * np.cos(angle) + \
              (yy - (self.pix_height - 1) / 2) * np.sin(angle)
        yct = (xx - (self.pix_width - 1) / 2) * np.sin(angle) - \
              (yy - (self.pix_height - 1) / 2) * np.cos(angle)
        rmaj = self.rad_lim * (self.r_eff.to(u.pc) / self.pxscale.to(u.pc)).value
        rmin = self.rad_lim * (self.r_eff.to(u.pc) / self.pxscale.to(u.pc)).value * self.ax_ratio
        mask = np.ones_like(brt, dtype=np.float32)
        rec = (xct ** 2 / rmaj ** 2) + (yct ** 2 / rmin ** 2) >= 1
        mask[rec] = 0
        mask = convolve_fft(mask, kernels.Gaussian2DKernel(3.), fill_value=0, allow_huge=True)
        brt = brt * mask
        brt = brt.reshape(self.pix_height, self.pix_width, 1)
        brt[brt < 0] = 0
        return brt

    def _get_2d_velocity(self):
        if hasattr(self, 'vel_rot') and (self.vel_rot is not None) and (self.vel_rot != 0):
            xx, yy = np.meshgrid(np.arange(self.pix_width), np.arange(self.pix_height))
            angle = (self.PA + 90 * u.degree).to(u.radian).value
            xct = (xx - (self.pix_width - 1) / 2) * np.cos(angle) + \
                  (yy - (self.pix_height - 1) / 2) * np.sin(angle)
            yct = (xx - (self.pix_width - 1) / 2) * np.sin(angle) - \
                  (yy - (self.pix_height - 1) / 2) * np.cos(angle)
            rad = np.sqrt(xct ** 2 + yct ** 2)
            vel_field = np.zeros_like(xx, dtype=np.float32) * velunit
            rec = rad > 0
            vel_field[rec] = self.vel_rot * np.sqrt(1 - self.ax_ratio ** 2) * xct[rec] / rad[rec]
            return vel_field
        else:
            return None



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
    _phi_bins: int = 90
    _theta_bins: int = 90
    _rad_bins: int = 0
    _npix_los: int = 100

    def __post_init__(self):
        self._assign_all_units()
        if self._rad_bins == 0:
            self._rad_bins = np.ceil(self.radius.to(u.pc).value / self.pxscale.to(u.pc).value * 3).astype(int)
        delta = np.round((len(self._cartesian_y_grid) - 1) / 2).astype(int)
        if (self.xc is not None) and (self.yc is not None):
            self.x0 = self.xc - delta
            self.y0 = self.yc - delta
        elif (self.x0 is not None) and (self.y0 is not None):
            self.xc = self.x0 + delta
            self.yc = self.y0 + delta
        self._ref_line_id = 0

    @cached_property
    def _theta_grid(self):
        return np.linspace(0, np.pi, self._theta_bins)

    @cached_property
    def _phi_grid(self):
        return np.linspace(0, 2 * np.pi, self._phi_bins)

    @cached_property
    def _rad_grid(self):
        return np.linspace(0, self.radius, self._rad_bins)

    @cached_property
    def _cartesian_z_grid(self):
        npix = np.ceil(1.02 * self.radius / self.pxscale).astype(int)
        return np.linspace(-npix, npix, 2 * npix + 1) * self.pxscale

    @cached_property
    def _cartesian_y_grid(self):
        return self._cartesian_z_grid.copy()

    @cached_property
    def _cartesian_x_grid(self):
        return np.linspace(-1.02, 1.02, self._npix_los) * self.radius

    @cached_property
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

            brt += np.nansum(Parallel(n_jobs=lvmdatasimulator.n_process)(delayed(brightness_inhomogeneities_sphere)
                                                                         (harm_amplitudes, ll, phi_cur, theta_cur,
                                                                         rho, med, self.radius, self.thickness)
                                                                         for ll in np.arange(1,
                                                                                             self.perturb_degree + 1)),
                             axis=0)
            brt[brt < 0] = 0
        if med > 0:
            brt = brt / np.nansum(brt)
        return brt

    @cached_property
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
            radius = np.insert(radius, 0, self.radius * (1 - self.thickness))
            fluxes = np.insert(fluxes, 0, fluxes[:, 0], axis=1)
            index_ha = np.flatnonzero(hdu[self.spectrum_id].data[1:, 0] == 6562.81)
            if self.n_brightest_lines is not None and \
                    (self.n_brightest_lines > 0) and (self.n_brightest_lines < len(fluxes)):
                indexes_sorted = np.argsort(hdu[self.spectrum_id].data[1:, 1])[::-1]
                fluxes = fluxes[indexes_sorted[:self.n_brightest_lines], :]
                index_ha = np.flatnonzero(hdu[self.spectrum_id].data[1:, 0][indexes_sorted] == 6562.81)
            if len(index_ha) == 1:
                self._ref_line_id = index_ha[0]

            brt = np.array(Parallel(n_jobs=lvmdatasimulator.n_process)(delayed(sphere_brt_in_line)
                                                                       (self._brightness_3d_spherical, rho,
                                                                        radius, flux)
                                                                       for flux in fluxes)).reshape((fluxes.shape[0],
                                                                                                     s[0], s[1], s[2]))
            return brt / np.nansum(brt[self._ref_line_id])

    @cached_property
    def _brightness_3d_cartesian(self):
        return interpolate_sphere_to_cartesian(self._brightness_3d_spherical, x_grid=self._cartesian_x_grid,
                                               y_grid=self._cartesian_y_grid, z_grid=self._cartesian_z_grid,
                                               rad_grid=self._rad_grid, theta_grid=self._theta_grid,
                                               phi_grid=self._phi_grid, pxscale=self.pxscale)

    @cached_property
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

    @cached_property
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

    @cached_property
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
        return (self._brightness_3d_cartesian[:, :, :, None] * (
                fluxunit / u.pc ** 3) * self._turbulent_lsf(velocity)).to(fluxunit / velunit / u.pc ** 3)

    @cached_property
    def line_profile(self) -> (fluxunit / velunit):
        """
        Produces the distribution of the observed line profiles in each pixels of the sky plane
        """
        vel_axis = self.spectral_axis.to(velunit, equivalencies=u.spectral())
        _, _, _, vels = np.meshgrid(self._cartesian_z_grid,
                                    self._cartesian_y_grid,
                                    self._cartesian_x_grid,
                                    vel_axis, indexing='ij')
        spectrum = (
                np.sum(self._d_spectrum_cartesian(vels), axis=2
                       ).T * (self._cartesian_x_grid[1] - self._cartesian_x_grid[0]
                              ) * (self._cartesian_y_grid[1] - self._cartesian_y_grid[0]
                                   ) * (self._cartesian_z_grid[1] - self._cartesian_z_grid[0])
        )
        return spectrum / np.sum(spectrum, axis=0)


@dataclass
class CustomNebula(Nebula):
    """
        Class defining the custom nebulae with the user-defined distribution of the brighntess, continuum
        and line shapes in different lines.
    """
    perturb_amplitude: float = 0
    brt_map: np.array = None
    lines_maps: np.array = None
    lines_wl: np.array = None


@dataclass
class ISM:
    """
    Class defining the ISM contribution to the field of view
    """
    wcs: WCS
    width: int = 401  # Width of field of view in pixels
    height: int = 401  # Width of field of view in pixels
    spec_resolution: u.Angstrom = 0.06 * u.Angstrom  # Spectral resolution of the simulation
    npix_line: int = 1  # Minimal number of pixels for a resolution element at wl = 10000A for construction of vel.grid
    distance: u.kpc = 50 * u.kpc  # Distance to the object for further conversion between arcsec and pc
    sys_velocity: velunit = 0 * velunit  # Systemic velocity to center the vel.grid on
    vel_amplitude: velunit = 150 * velunit  # Maximal deviation from the systemic velocity to setup vel.grid
    turbulent_sigma: velunit = 20. * velunit  # turbulence vel. disp. to be used for every nebula unless other specified
    R_V: float = 3.1  # R_V value defining the reddening curve (to be used unless other value is provided for a nebula)
    ext_law: str = 'F99'  # Reddening law (to be used unless other value is provided for a nebula)

    def __post_init__(self):
        assign_units(self, ['vel_amplitude', 'turbulent_sigma', 'sys_velocity', 'distance', 'spec_resolution'],
                     [velunit, velunit, velunit, u.kpc, u.Angstrom])
        self.content = fits.HDUList()
        self.content.append(fits.PrimaryHDU(header=self.wcs.to_header(), data=np.zeros(shape=(self.height, self.width),
                                                                                       dtype=np.float32)))
        self.vel_grid = np.linspace(-self.vel_amplitude + self.sys_velocity,
                                    self.vel_amplitude + self.sys_velocity,
                                    np.ceil(self.vel_amplitude / self.vel_resolution).astype(int) * 2 + 1)
        self.pxscale = proj_plane_pixel_scales(self.wcs)[0] * 3600 * self.distance.to(u.pc) / 206265.
        self.content[0].header['width'] = (self.width, "Width of field of view, px")
        self.content[0].header['height'] = (self.height, "Height of field of view, px")
        self.content[0].header['PhysRes'] = (self.pxscale.value, "Physical resolution, pc/px")
        self.content[0].header['Dist'] = (self.distance.value, "Distance, kpc")
        self.content[0].header['Vsys'] = (self.distance.value, "Systemic Velocity, km/s")
        self.content[0].header['VelRes'] = (self.vel_resolution.value, "Velocity resolution, km/s/px")
        self.content[0].header['TurbSig'] = (self.turbulent_sigma.value, "Default turbulent velocity dispersion, km/s")
        self.content[0].header['Nobj'] = (0, "Number of generated nebulae")

    @cached_property
    def vel_resolution(self):
        return (self.spec_resolution / self.npix_line / (10000 * u.Angstrom) * c.c).to(velunit)

    def _get_continuum(self, my_comp, wl_grid):
        """
        Properly extracts continuum for current nebula taking into account its shape and surface brightness
        :param my_comp:
        :param wl_grid:
        :return: continuum
        """
        cont_type = self.content[my_comp + "_CONTINUUM"].header.get("CONTTYPE")
        continuum = self.content[my_comp + "_CONTINUUM"].data
        cont_norm = self.content[my_comp + "_CONTINUUM"].header.get("CONTFLUX")
        cont_norm_wl = self.content[my_comp + "_CONTINUUM"].header.get("CONTWL")
        if cont_type.lower() == 'model':
            cont_wl_fullrange = continuum[0, :]
            cont_fullrange = continuum[1, :]
            p = interp1d(continuum[0, :], continuum[1, :], assume_sorted=True,
                         bounds_error=False, fill_value='extrapolate')
            continuum = p(wl_grid)
        elif cont_type.lower() == 'poly':
            p = np.poly1d(continuum)
            cont_wl_fullrange = np.linspace(3500, 10000, 501)
            cont_fullrange = p(cont_wl_fullrange)
            continuum = p(wl_grid)
        elif cont_type.lower() == 'bb':
            cont_wl_fullrange = np.linspace(3500, 10000, 501)
            cont_fullrange = 1 / cont_wl_fullrange ** 5 / (
                    np.exp(6.63e-27 * 3e10 / cont_wl_fullrange / 1e-8 / continuum / 1.38e-16) - 1)
            continuum = 1 / wl_grid ** 5 / (np.exp(6.63e-27 * 3e10 / wl_grid / 1e-8 / continuum / 1.38e-16) - 1)

        t_filter = None
        if type(cont_norm_wl) == str:
            file_filter = os.path.join(lvmdatasimulator.ROOT_DIR, 'data', 'instrument', 'filters', cont_norm_wl+".dat")
            if not os.path.isfile(file_filter):
                log.warning("Cannot find filter {0}. "
                            "Default Wavelength = 5500A will be used for continuum normalization".format(cont_norm_wl))
                cont_norm_wl = 5500.
                t_filter = None
            else:
                t_filter = ascii.read(file_filter, names=['lambda', 'transmission'])
                cont_norm_wl = np.sum(t_filter['lambda'] * t_filter['transmission']) / np.sum(t_filter['transmission'])
        if t_filter is None:
            cont_model_max = cont_fullrange[np.argmin(abs(cont_wl_fullrange - cont_norm_wl))]
        else:
            dl = np.roll(t_filter['lambda'], -1) - t_filter['lambda']
            dl[-1] = dl[-2]
            w_filter = np.sum(dl * t_filter['transmission'])/np.max(t_filter['transmission'])
            p = interp1d(t_filter['lambda'], t_filter['transmission'], assume_sorted=True,
                         fill_value=0, bounds_error=False)
            cont_model_max = np.sum(cont_fullrange * p(cont_wl_fullrange)) / w_filter
        if ~np.isfinite(cont_norm) or cont_norm <= 0:
            cont_norm = self.content[my_comp + "_CONTINUUM"].header.get("CONTMAG") * u.ABmag
            cont_norm = cont_norm.to(u.STmag, u.spectral_density(cont_norm_wl * u.AA)).to(u.erg/u.s/u.cm**2/u.AA).value
        return continuum / cont_model_max * cont_norm * (wl_grid[1] - wl_grid[0])

    def _add_fits_extension(self, name, value, obj_to_add, zorder=0, cur_wavelength=0, add_fits_kw=None,
                            add_counter=False):
        self.content.append(fits.ImageHDU(np.atleast_1d(value), name=name))
        self.content[-1].header['Nebtype'] = (type(obj_to_add).__name__, "Type of the nebula")
        is_dark = ((obj_to_add.max_brightness <= 0) and (obj_to_add.max_extinction > 0))
        self.content[-1].header['Dark'] = (is_dark, " Emitting or absorbing nebula?")
        self.content[-1].header['X0'] = (obj_to_add.x0, "Position in the field of view")
        self.content[-1].header['Y0'] = (obj_to_add.y0, "Position in the field of view")
        self.content[-1].header['Zorder'] = (zorder, "Z-order in the field of view")
        self.content[-1].header['NCHUNKS'] = (obj_to_add.nchunks, "N of chunks for convolution")
        if type(obj_to_add) in [Bubble, Cloud]:
            self.content[-1].header['Radius'] = (obj_to_add.radius.to_value(u.pc), "Radius of the nebula, pc")
            self.content[-1].header['PertOrd'] = (obj_to_add.perturb_degree, "Degree to produce random perturbations")
            self.content[-1].header['PertAmp'] = (obj_to_add.perturb_amplitude, "Max amplitude of random perturb.")
        if type(obj_to_add) in [DIG, Nebula, Rectangle, Ellipse, Circle]:
            self.content[-1].header['PertScl'] = (obj_to_add.perturb_scale.to_value(u.pc),
                                                  "Scale of the random perturbations, pc")
            self.content[-1].header['PertAmp'] = (obj_to_add.perturb_amplitude, "Max amplitude of random perturb.")
        if type(obj_to_add) in [Filament]:
            self.content[-1].header['Width'] = (obj_to_add.width.to_value(u.pc), 'Width of the filament, pc')
            self.content[-1].header['Length'] = (obj_to_add.length.to_value(u.pc), 'Length of the filament, pc')

        if type(obj_to_add) not in [Bubble, Cloud, Galaxy]:
            if obj_to_add.vel_gradient is not None:
                self.content[-1].header['Vgrad'] = (obj_to_add.vel_gradient.to_value(velunit / u.pc),
                                                    'Velocity gradient, km/s per pc')
                if obj_to_add.vel_pa is not None:
                    self.content[-1].header['PAkin'] = (obj_to_add.vel_pa.to_value(u.degree),
                                                        'Kinematical PA, degree')
        if type(obj_to_add) in [Nebula, Rectangle]:
            self.content[-1].header['Width'] = (obj_to_add.width.to_value(u.pc), 'Width of the nebula, pc')
            self.content[-1].header['Height'] = (obj_to_add.height.to_value(u.pc), 'Height of the nebula, pc')
        if type(obj_to_add) in [Ellipse, Filament, Galaxy]:
            self.content[-1].header['PA'] = (obj_to_add.PA.to_value(u.degree), 'Position angle, degree')
        if type(obj_to_add) in [Galaxy]:
            self.content[-1].header['Reff'] = (obj_to_add.r_eff.to_value(u.kpc), 'Effective radius of the galaxy, kpc')
            self.content[-1].header['NSersic'] = (obj_to_add.n, 'Sersic index')
            self.content[-1].header['Rlim'] = (obj_to_add.rad_lim, 'Limiting distance in Reff')
            self.content[-1].header['Vrot'] = (obj_to_add.vel_rot.to_value(velunit),
                                               'Rotational velocity, km/s')
            self.content[-1].header['PAkin'] = (obj_to_add.vel_pa.to_value(u.degree),
                                                'Kinematical PA, degree')
        if type(obj_to_add) in [Ellipse, Circle]:
            self.content[-1].header['Radius'] = (obj_to_add.radius.to_value(u.kpc), 'Radius (major axis), pc')
        if type(obj_to_add) in [Ellipse, Galaxy]:
            self.content[-1].header['AxRat'] = (obj_to_add.ax_ratio, "Axis ratio")
        if (obj_to_add.max_brightness <= 0) and (obj_to_add.max_extinction > 0):
            self.content[-1].header['MaxExt'] = (obj_to_add.max_extinction.value, "Max extinction, mag/pix")
        elif obj_to_add.max_brightness > 0:
            self.content[-1].header['MaxBrt'] = (obj_to_add.max_brightness.value, "Max brightness, erg/s/cm^2/arcsec^2")
        if type(obj_to_add) == Bubble:
            self.content[-1].header['Vexp'] = (obj_to_add.expansion_velocity.to_value(velunit),
                                               'Expansion velocity, km/s')
        self.content[-1].header['SysVel'] = (obj_to_add.sys_velocity.to_value(velunit), "Systemic velocity, km/s")
        self.content[-1].header['TurbVel'] = (obj_to_add.turbulent_sigma.to_value(velunit),
                                              "ISM Velocity dispersion, km/s")
        self.content[-1].header['SpecID'] = (obj_to_add.spectrum_id, "Ref. spectrum ID in model grid")
        self.content[-1].header['NLines'] = (obj_to_add.n_brightest_lines, "Maximal number of lines to use")
        if cur_wavelength:
            self.content[-1].header['Lambda'] = (cur_wavelength, "Current line wavelength")
        if add_fits_kw is not None:
            for kw in add_fits_kw:
                self.content[-1].header[kw] = add_fits_kw[kw]
        if add_counter:
            self.content[0].header['Nobj'] = (self.content[0].header['Nobj'] + 1, "Total number of nebulae")

    def add_nebula(self, obj_to_add, obj_id=0, zorder=0, add_fits_kw=None, continuum=None):
        """
        Method to add the particular nebula to the ISM object and to the output multi-extensions fits file
        """
        if type(obj_to_add) not in [Nebula, Bubble, Filament, DIG, Cloud, Galaxy, Ellipse, Circle,
                                    Rectangle, CustomNebula]:
            log.warning('Skip nebula of wrong type ({0})'.format(type(obj_to_add)))
            return

        if (obj_to_add.max_brightness <= 0) and (obj_to_add.max_extinction <= 0) and (continuum is None):
            log.warning('Skip nebula with zero extinction and brightness')
            return
        if type(obj_to_add) == CustomNebula:
            brt = obj_to_add.brt_map
            brt_4d = obj_to_add.lines_maps
        else:
            if obj_to_add.max_brightness > 0:
                brt = obj_to_add.brightness_skyplane.value
                if obj_to_add.spectrum_id is not None and not obj_to_add.linerat_constant:
                    brt_4d = obj_to_add.brightness_skyplane_lines.value
                else:
                    brt_4d = None
            elif obj_to_add.max_extinction > 0:
                brt = obj_to_add.extinction_skyplane.value
                brt_4d = None
            elif continuum is not None:
                brt = obj_to_add.brightness_skyplane
                brt_4d = None

        self._add_fits_extension(name="Comp_{0}_Brightness".format(obj_id), value=brt,
                                 obj_to_add=obj_to_add, zorder=zorder, add_fits_kw=add_fits_kw, add_counter=True)
        if obj_to_add.max_brightness > 0:
            if brt_4d is not None:
                if type(obj_to_add) == CustomNebula:
                    wl_list = obj_to_add.lines_wl
                    if obj_to_add.n_brightest_lines is not None and \
                            (obj_to_add.n_brightest_lines > 0) and (obj_to_add.n_brightest_lines < len(wl_list)):
                        wl_list = wl_list[np.argsort(np.max(brt_4d, axis=(1, 2)))[::-1][: obj_to_add.n_brightest_lines]]
                else:
                    if obj_to_add.spectrum_type == 'cloudy':
                        filename = lvmdatasimulator.CLOUDY_MODELS
                    elif obj_to_add.spectrum_type == 'mappings':
                        filename = lvmdatasimulator.MAPPINGS_MODELS
                    with fits.open(filename) as hdu:
                        wl_list = hdu[obj_to_add.spectrum_id].data[1:, 0]
                        if obj_to_add.n_brightest_lines is not None and \
                                (obj_to_add.n_brightest_lines > 0) and (obj_to_add.n_brightest_lines < len(wl_list)):
                            wl_list = wl_list[np.argsort(hdu[obj_to_add.spectrum_id].data[1:, 1]
                                                        )[::-1][: obj_to_add.n_brightest_lines]]

                for line_ind in range(brt_4d.shape[0]):
                    self._add_fits_extension(name="Comp_{0}_Flux_{1}".format(obj_id, wl_list[line_ind]),
                                             value=brt_4d[line_ind],
                                             obj_to_add=obj_to_add, zorder=zorder, add_fits_kw=add_fits_kw,
                                             cur_wavelength=wl_list[line_ind])
            elif obj_to_add.spectrum_id is not None and obj_to_add.linerat_constant:
                if obj_to_add.spectrum_type == 'cloudy':
                    filename = lvmdatasimulator.CLOUDY_MODELS
                elif obj_to_add.spectrum_type == 'mappings':
                    filename = lvmdatasimulator.MAPPINGS_MODELS

                with fits.open(filename) as hdu:
                    data_save = hdu[obj_to_add.spectrum_id].data[1:, :2]
                    if obj_to_add.n_brightest_lines is not None and \
                            (obj_to_add.n_brightest_lines > 0) and \
                            (obj_to_add.n_brightest_lines < len(hdu[obj_to_add.spectrum_id].data[1:, 0])):
                        data_save = data_save[np.argsort(data_save[:, 1])[::-1][: obj_to_add.n_brightest_lines]]
                    self._add_fits_extension(name="Comp_{0}_FluxRatios".format(obj_id),
                                             value=data_save.T,
                                             obj_to_add=obj_to_add, zorder=zorder, add_fits_kw=add_fits_kw)

        if type(obj_to_add) == Bubble:
            self._add_fits_extension(name="Comp_{0}_LineProfile".format(obj_id), value=obj_to_add.line_profile.value,
                                     obj_to_add=obj_to_add, zorder=zorder, add_fits_kw=add_fits_kw)

        if obj_to_add.vel_field is not None:
            self._add_fits_extension(name="Comp_{0}_Vel".format(obj_id), value=obj_to_add.vel_field.value,
                                     obj_to_add=obj_to_add, zorder=zorder, add_fits_kw=add_fits_kw)

        if continuum is not None:
            self._add_fits_extension(name="Comp_{0}_Continuum".format(obj_id), value=continuum,
                                     obj_to_add=obj_to_add, zorder=zorder, add_fits_kw=add_fits_kw)
        return self.content

    def save_ism(self, filename):
        total_map = None
        if self.content[0].header['Nobj'] > 0:
            total_map = self.get_map(wavelength=[6560., 6565.])
        if total_map is None:
            total_map = np.zeros(shape=(self.height, self.width), dtype=np.float32)
        self.content[0].data = total_map
        self.content[0].header['IM_WL'] = ('6560-6565', "Rest-frame wavelength range for the ref. image")
        self.content.writeto(filename, overwrite=True)
        log.info("Generated ISM saved to {0}".format(filename))

    def process_custom_nebula(self, params, xc=None, yc=None, model_index=None):
        """
        Processes the input parameters for custom nebula object

        Args:
            params: dict -- contains keywords and values for a nebula properties
            xc: int -- pixel X coordinate of the center of the nebula in FOV
                         (calculated in 'generate' method based on provided offsets)
            yc: int -- pixel Y coordinate of the center of the nebula in FOV
                         (calculated in 'generate' method based on provided offsets)
            model_index: int or None -- ID of cloudy models to use if maps in different lines are not present
        Returns:
             CustomNebula object, or None if something went wrong
        """

        pxsize = params['pxsize']
        try:
            pxsize = pxsize << u.pc
        except UnitConversionError:
            try:
                pxsize = pxsize << u.arcsec
            except UnitConversionError:
                log.error(f"Wrong unit for parameter 'pxsize' - should be angle or parsec")
                return None
            pxsize = (pxsize.value * params['distance'].to_value(u.pc) / 206265.) << u.pc
        pxsize_out = self.pxscale * (params['distance'].to_value(u.pc) / self.distance.to_value(u.pc))
        pxsize = pxsize.value
        pxsize_out = pxsize_out.value

        # cut arrays if they have even size along axes
        xc_input = (params['brightness_map'].shape[1]) // 2
        yc_input = (params['brightness_map'].shape[0]) // 2
        nlines = 0
        lines_map = {}

        if 'brightness_lines' in params and isinstance(params['brightness_lines'], dict):
            for k, v in params['brightness_lines'].items():
                nlines += 1
                if v.shape != params['brightness_map'].shape:
                    log.error("Size of all arrays containing the brightness in each lines "
                              "should be equal to that of the `brightness_map'")
                    return None
                lines_map[k] = params['brightness_lines'][k][
                               params['brightness_map'].shape[0] - 2 * yc_input:,
                               params['brightness_map'].shape[1] - 2 * xc_input:]
                lines_map[k][~np.isfinite(lines_map[k]) | (lines_map[k] < 0)] = 0
        brt_map = params['brightness_map'][
                  params['brightness_map'].shape[0] - 2 * yc_input:, params['brightness_map'].shape[1] - 2 * xc_input:]
        brt_map[~np.isfinite(brt_map) | (brt_map < 0)] = 0
        if np.isclose(pxsize_out, pxsize):
            shape_out = brt_map.shape
        else:
            shape_out = (np.ceil(yc_input*pxsize/pxsize_out).astype(int) * 2 + 1,
                         np.ceil(xc_input*pxsize/pxsize_out).astype(int) * 2 + 1)
        xc_out = shape_out[1] // 2
        yc_out = shape_out[0] // 2
        if nlines == 0:
            wavelengths = None
        else:
            wavelengths = np.array([])
        brt_maps_out = np.zeros(shape=(1 + nlines, shape_out[0], shape_out[1]), dtype=np.float32)
        brt_max = np.nanmax(brt_map)
        if shape_out == brt_map.shape:
            brt_maps_out[0] = brt_map
            if nlines > 0:
                ind = 1
                for k, v in lines_map.items():
                    wavelengths = np.append(wavelengths, k)
                    brt_maps_out[ind] = v
                    ind += 1
        else:
            # Interpolate brightness distributions to the ISM grid:
            brt_maps = brt_map.reshape((1, brt_map.shape[0], brt_map.shape[1]))

            if nlines > 0:
                for k, v in lines_map.items():
                    wavelengths = np.append(wavelengths, k)
                    brt_maps = np.append(brt_maps, v)
            brt_maps = brt_maps.reshape((1+nlines, brt_map.shape[0],
                                         brt_map.shape[1]))

            xx, yy = np.arange(brt_maps.shape[2]), np.arange(brt_maps.shape[1])
            xx_out, yy_out = np.meshgrid(np.arange(shape_out[1]), np.arange(shape_out[0]))
            xx_out = (xx_out - xc_out) * pxsize_out / pxsize + xc_input
            yy_out = (yy_out - yc_out) * pxsize_out / pxsize + yc_input
            for ind, brt_slice in enumerate(brt_maps):
                p = RectBivariateSpline(xx, yy, brt_slice.T, kx=1, ky=1)
                output = np.zeros_like(xx_out, dtype=np.float32)
                rec = (xx_out >= 0) & (xx_out <= (brt_maps.shape[2]-1)) & \
                      (yy_out >= 0) & (yy_out <= (brt_maps.shape[1]-1))
                output[rec] = p.ev(xx_out[rec], yy_out[rec])
                brt_maps_out[ind] = output.reshape(shape_out)
        if nlines == 0:
            lines_maps = None
        else:
            lines_maps = brt_maps_out[1:]
            model_index = None
        neb = CustomNebula(xc=xc, yc=yc, max_brightness=brt_max, turbulent_sigma=params['turbulent_sigma'],
                           sys_velocity=params['sys_velocity'], vel_gradient=params['vel_gradient'],
                           vel_pa=params['vel_pa'], spectrum_id=model_index,
                           n_brightest_lines=params['n_brightest_lines'],
                           pxscale=self.pxscale * (params['distance'].to(u.pc) / self.distance.to(u.pc)),
                           pix_width=shape_out[1], pix_height=shape_out[0],
                           brt_map=brt_maps_out[0], lines_maps=lines_maps, lines_wl=wavelengths)
        return neb

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
                                max_brightness: 1e-16 * u.erg / u.cm**2 / u.s / u.arcsec ** 2,
                                RA: "08h12m13s",
                                DEC: "-20d14m13s",
                                'perturb_degree': 8, # max. order of spherical harmonics to generate inhomogeneities
                                'perturb_amplitude': 0.1, # relative max. amplitude of inhomogeneities,
                                'perturb_scale': 200 * u.pc,  # spatial scale to generate inhomogeneities (DIG only)
                                'distance': 50 * u.kpc,  # distance to the nebula (default is from ISM)
                                'cloudy_id': None,
                                'cloudy_params': {'Z': 0.5, 'qH': 49., 'nH': 10, 'Teff': 30000, 'Geometry': 'Sphere'},
                                'model_id': None,  # id of pre-generated model
                                'model_params': {'Z': 0.5, 'qH': 49., 'nH': 10, 'Teff': 30000, 'Geometry': 'Sphere'}, #
                                    parameters defining the pre-generated Cloudy model (used if spectrum_id is None)
                                'model_type: 'cloudy',
                                'n_brightest_lines': 10,  # only this number of the brightest lines will be evaluated
                                'linerat_constant': False #  if True -> lines ratios don't vary across Cloud/Bubble
                                'continuum_type': 'BB' or 'Model' or 'Poly' # type of the continuum model
                                'continuum_data': model_id or [poly_coefficients] or Teff or dict with "wl" and "flux"
                                                # value defining cont. shape
                                'continuum_flux': 1e-16 * u.erg / u.cm ** 2 / u.s / u.arcsec **2 / u.AA,
                                'continuum_mag': 22 * u.mag,
                                'continuum_wl': 5500, # could be also R, V, B,
                                'ext_law': 'F99',  # Extinction law, one of those used by pyneb (used for dark nebulae)
                                'ext_rv': 3.1,  # Value of R_V for extinction curve calculation (used for dark nebulae)
                                'vel_gradient: 12. * u.km / u.s / u.pc  # Line-of-sight velocity gradient
                                'vel_pa: 30. * u.degree  # PA of kinematical axis (for vel_gradient or vel_rot)
                                }]
        """
        if type(all_objects) is dict:
            all_objects = [all_objects]
        if type(all_objects) not in [list, tuple]:
            log.warning('Cannot generate nebulae as the input is not a list or tuple')
            return None
        all_objects = [cobj for cobj in all_objects if cobj.get('type') in ['Nebula', 'Bubble', 'Galaxy',
                                                                            'Filament', 'DIG', 'Cloud',
                                                                            'Rectangle', 'Circle', 'Ellipse',
                                                                            'CustomNebula']]
        n_objects = len(all_objects)
        log.info("Start generating {} nebulae".format(n_objects))
        bar = progressbar.ProgressBar(max_value=n_objects).start()
        obj_id = self.content[0].header['Nobj']
        obj_id_ini = self.content[0].header['Nobj']
        for ind_obj, cur_obj in enumerate(all_objects):
            bar.update(ind_obj)
            # Setup default parameters for missing keywords
            kin_pa_default = 0
            if 'PA' in cur_obj:
                kin_pa_default = cur_obj['PA']
            for k, v, unit in zip(['max_brightness', 'max_extinction', 'thickness',
                                   'expansion_velocity', 'sys_velocity',
                                   'turbulent_sigma', 'perturb_degree',
                                   'perturb_amplitude', 'perturb_scale', 'radius', 'distance',
                                   'continuum_type', 'continuum_data', 'continuum_flux', 'continuum_mag',
                                   'continuum_wl', 'ext_law', 'ext_rv', 'vel_gradient', 'vel_rot', 'vel_pa',
                                   'n_brightest_lines', 'offset_RA', 'offset_DEC', 'RA', 'DEC',
                                   'pxsize'],
                                  [0, 0, 1., 0, self.sys_velocity, self.turbulent_sigma, 0, 0.1, 0, 0, self.distance,
                                   None, None, 0, None, 5500., self.ext_law, self.R_V, 0, 0, kin_pa_default, None,
                                   None, None, None, None, self.pxscale],
                                  [fluxunit, u.mag, None, velunit, velunit, velunit, None, None,
                                   u.pc, u.pc, u.kpc, None, None, fluxunit/u.AA, None, u.Angstrom,
                                   None, None, velunit / u.pc, velunit, u.degree, None, u.arcsec, u.arcsec,
                                   u.degree, u.degree, None]):
                set_default_dict_values(cur_obj, k, v, unit=unit)

            for k in ['max_brightness', 'max_extinction', 'radius', 'continuum_flux']:
                if cur_obj[k] < 0:
                    cur_obj[k] = 0

            if cur_obj['type'] == 'CustomNebula':
                if not isinstance(cur_obj.get('brightness_map'), np.ndarray):
                    log.warning("Wrong set of parameters for the nebula #{0}: skip this one".format(ind_obj))
                    continue
                cur_obj['max_brightness'] = 1.

            if (cur_obj['max_brightness'] == 0) and (cur_obj['max_extinction'] == 0) and \
                    (((cur_obj['continuum_mag'] is None) and (cur_obj['continuum_flux'] == 0)) or
                     (cur_obj['continuum_data'] is None) or (cur_obj['continuum_type'] is None)):
                log.warning("Wrong set of parameters for the nebula #{0}: skip this one".format(ind_obj))
                continue

            model_type = cur_obj.get('model_type', 'cloudy')  # define the model type before continuing

            if cur_obj.get('cloudy_id', None) is not None:
                warnings.warn('Cloudy_id will be removed in the next versions of the code.'
                              'Please use "model_id" and "model_type: cloudy" instead.', DeprecationWarning)

                cur_obj['model_id'] = cur_obj.get('cloudy_id')
                cur_obj['model_type'] = 'cloudy'

            if cur_obj.get('cloudy_params', None) is not None:
                warnings.warn('Cloudy_params will be removed in the next versions of the code.'
                              'Please use "model_params" and "model_type: cloudy" instead.', DeprecationWarning)

                cur_obj['model_params'] = cur_obj.get('cloudy_params')
                cur_obj['model_type'] = 'cloudy'



            # define which default models to use
            if model_type == 'cloudy':
                id_def = lvmdatasimulator.CLOUDY_SPEC_DEFAULTS['id']
                param_def = lvmdatasimulator.CLOUDY_SPEC_DEFAULTS
                model_file = lvmdatasimulator.CLOUDY_MODELS
            elif model_type == 'mappings':
                id_def = lvmdatasimulator.MAPPINGS_SPEC_DEFAULTS['id']
                param_def = lvmdatasimulator.MAPPINGS_SPEC_DEFAULTS
                model_file = lvmdatasimulator.MAPPINGS_MODELS
            else:
                raise ValueError(f'{model_type} models are not supported')

            # if no emission is wanted, do not set any model
            if cur_obj['max_brightness'] <= 0:
                model_index = None
                model_id = None
                # if no model is selected, go for the standard one
            elif cur_obj.get('model_id') is None and type(cur_obj.get('model_params')) is not dict:

                if not (cur_obj['type'] == 'CustomNebula' and
                        isinstance(cur_obj.get('brightness_lines'), dict)):

                    log.warning("No model ids or model parameters are set for the nebula #{0}: "
                        "use default {1} 'model_id={2}'".format(ind_obj, model_type, id_def))

                cur_obj['model_id'] = id_def

                model_index, model_id = find_model_id(file=model_file,
                                                        check_id=cur_obj.get('model_id'),
                                                        params=cur_obj.get('model_params'),
                                                        defaults=id_def)


            elif cur_obj.get('model_id') is None and (type(cur_obj.get('model_params')) is dict):
                for p in param_def:
                    if p == 'id':
                        continue
                    if cur_obj['model_params'].get(p) is None:
                        cur_obj['model_params'][p] = param_def[p]

                model_index, model_id = find_model_id(file=model_file,
                                                        check_id=cur_obj.get('model_id'),
                                                        params=cur_obj.get('model_params'),
                                                        defaults=param_def['id'])
            elif cur_obj.get('model_id'):

                model_index, model_id = find_model_id(file=model_file,
                                                        check_id=cur_obj.get('model_id'),
                                                        params=cur_obj.get('model_params'),
                                                        defaults=param_def['id'])

            # if lvmdatasimulator.CLOUDY_MODELS is None or (cur_obj['max_brightness'] <= 0):
            #     cloudy_model_index = None
            #     model_id = None
            # else:
            #     if cur_obj.get('cloudy_id') is None:
            #         if cur_obj.get('cloudy_params') is None or (type(cur_obj.get('cloudy_params')) is not dict):
            #             if not (cur_obj['type'] == 'CustomNebula' and
            #                     isinstance(cur_obj.get('brightness_lines'), dict)):
            #                 log.warning("Neither of 'cloudy_id' or 'cloudy_params' is set for the nebula #{0}: "
            #                             "use default 'cloudy_id={1}'".format(ind_obj,
            #                                                                  lvmdatasimulator.CLOUDY_SPEC_DEFAULTS['id']))
            #             cur_obj['cloudy_id'] = lvmdatasimulator.CLOUDY_SPEC_DEFAULTS['id']
            #         else:
            #             for p in lvmdatasimulator.CLOUDY_SPEC_DEFAULTS:
            #                 if p == 'id':
            #                     continue
            #                 if cur_obj['cloudy_params'].get(p) is None:
            #                     cur_obj['cloudy_params'][p] = lvmdatasimulator.CLOUDY_SPEC_DEFAULTS[p]

            #     cloudy_model_index, model_id = find_model_id(file=lvmdatasimulator.CLOUDY_MODELS,
            #                                                         check_id=cur_obj.get('cloudy_id'),
            #                                                         params=cur_obj.get('cloudy_params'))

            if cur_obj.get('linerat_constant') is None:
                if cur_obj['type'] in ['Bubble', 'Cloud']:
                    cur_obj['linerat_constant'] = False
                else:
                    cur_obj['linerat_constant'] = True

            if cur_obj['type'] == 'CustomNebula':
                if 'brightness_lines' in cur_obj and isinstance(cur_obj['brightness_lines'], dict):
                    cur_obj['linerat_constant'] = False
                else:
                    cur_obj['linerat_constant'] = True

            if cur_obj['type'] == 'DIG':
                if cur_obj.get('max_brightness') is None or cur_obj.get('max_brightness') <= 0:
                    log.warning("Wrong set of parameters for the nebula #{0}: skip this one".format(ind_obj))
                    continue
                if not cur_obj.get('zorder'):
                    cur_obj['zorder'] = -1
                if cur_obj['perturb_scale'] < 0:
                    cur_obj['perturb_scale'] = 0
                generated_object = DIG(max_brightness=cur_obj.get('max_brightness'),
                                       turbulent_sigma=cur_obj['turbulent_sigma'],
                                       sys_velocity=cur_obj['sys_velocity'],
                                       vel_gradient=cur_obj['vel_gradient'],
                                       vel_pa=cur_obj['vel_pa'],
                                       spectrum_id=model_index,
                                       spectrum_type=model_type,
                                       n_brightest_lines=cur_obj['n_brightest_lines'],
                                       pxscale=self.pxscale * (cur_obj['distance'].to(u.pc) / self.distance.to(u.pc)),
                                       perturb_scale=cur_obj['perturb_scale'],
                                       perturb_amplitude=cur_obj['perturb_amplitude'],
                                       pix_width=self.width, pix_height=self.height,
                                       )
            else:
                # ==== Check input parameters and do necessary conversions
                if not cur_obj.get('zorder'):
                    cur_obj['zorder'] = 0
                if not ((cur_obj.get('RA') is not None and cur_obj.get('DEC') is not None) or
                        (cur_obj.get('X') is not None and cur_obj.get('Y') is not None) or
                        (cur_obj.get('offset_X') is not None and cur_obj.get('offset_Y') is not None) or
                        (cur_obj.get('offset_RA') is not None and cur_obj.get('offset_DEC') is not None)):
                    log.warning("Wrong set of parameters for the nebula #{0}: skip this one".format(ind_obj))
                    continue
                if cur_obj['type'] in ['Rectangle', 'Nebula'] and not (('width' in cur_obj) and ('height' in cur_obj)):
                    log.warning("Wrong set of parameters for the nebula #{0}: skip this one".format(ind_obj))
                    continue
                if cur_obj['type'] in ["Bubble", "Cloud"] and (model_type == 'mappings'):
                    log.warning("Mappings models are not spatially resolved. Skip object #{}.".format(ind_obj))
                    continue
                if (cur_obj['type'] in ["Bubble", "Cloud", "Ellipse", 'Circle']) and (cur_obj['radius'] == 0):
                    log.warning("Wrong set of parameters for the nebula #{0}: skip this one".format(ind_obj))
                    continue
                if cur_obj['type'] == 'Filament' and not (('length' in cur_obj) and ('PA' in cur_obj)):
                    log.warning("Wrong set of parameters for the nebula #{0}: skip this one".format(ind_obj))
                    continue
                if cur_obj['type'] == 'Galaxy' and not (('r_eff' in cur_obj) and ('PA' in cur_obj) and
                                                        ('ax_ratio' in cur_obj)):
                    log.warning("Wrong set of parameters for the nebula #{0}: skip this one".format(ind_obj))
                    continue
                if cur_obj['type'] == 'Galaxy':
                    if 'n' not in cur_obj:
                        log.info("Set default Sersic index n=1 for the nebula #{0}".format(ind_obj))
                        cur_obj['n'] = 1
                    if 'rad_lim' not in cur_obj:
                        cur_obj['rad_lim'] = 3.

                if not (cur_obj.get('X') is not None and cur_obj.get('Y') is not None):
                    if cur_obj.get('offset_X') is not None and cur_obj.get('offset_Y') is not None:
                        x = (self.width - 1) / 2. + cur_obj.get('offset_X')
                        y = (self.height - 1) / 2. + cur_obj.get('offset_Y')
                    elif cur_obj.get('RA') is not None and cur_obj.get('DEC') is not None:
                        radec = SkyCoord(ra=cur_obj.get('RA'), dec=cur_obj.get('DEC'))
                        x, y = self.wcs.world_to_pixel(radec)
                    elif cur_obj.get('offset_RA') is not None and cur_obj.get('offset_DEC') is not None:
                        x = (self.width - 1) / 2. - (cur_obj.get('offset_RA').to_value(u.degree) /
                                                     proj_plane_pixel_scales(self.wcs)[0])
                        y = (self.height - 1) / 2. + (cur_obj.get('offset_DEC').to_value(u.degree) /
                                                      proj_plane_pixel_scales(self.wcs)[0])
                    x = np.round(x).astype(int)
                    y = np.round(y).astype(int)
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
                    log.info("Set default width of the filament 0.1 pc for the nebula #{0}".format(ind_obj))
                    cur_obj['width'] = 0.1 * u.pc

                # ==== Start calculations of different nebulae
                if cur_obj['type'] == "Bubble":
                    generated_object = Bubble(xc=x, yc=y,
                                              max_brightness=cur_obj.get('max_brightness'),
                                              max_extinction=cur_obj.get('max_extinction'),
                                              spectral_axis=self.vel_grid,
                                              expansion_velocity=cur_obj.get('expansion_velocity'),
                                              thickness=cur_obj['thickness'],
                                              radius=cur_obj['radius'],
                                              pxscale=self.pxscale * (cur_obj['distance'].to(u.pc) /
                                                                      self.distance.to(u.pc)),
                                              perturb_degree=cur_obj['perturb_degree'],
                                              perturb_amplitude=cur_obj['perturb_amplitude'],
                                              turbulent_sigma=cur_obj['turbulent_sigma'],
                                              sys_velocity=cur_obj['sys_velocity'],
                                              spectrum_id=model_index,
                                              spectrum_type=model_type,
                                              n_brightest_lines=cur_obj['n_brightest_lines'],
                                              linerat_constant=cur_obj['linerat_constant'],
                                              )
                elif cur_obj['type'] == "Cloud":
                    generated_object = Cloud(xc=x, yc=y,
                                             max_brightness=cur_obj.get('max_brightness'),
                                             max_extinction=cur_obj.get('max_extinction'),
                                             thickness=cur_obj['thickness'],
                                             radius=cur_obj['radius'],
                                             pxscale=self.pxscale * (cur_obj['distance'].to(u.pc) /
                                                                     self.distance.to(u.pc)),
                                             perturb_degree=cur_obj['perturb_degree'],
                                             perturb_amplitude=cur_obj['perturb_amplitude'],
                                             spectrum_id=model_index,
                                             spectrum_type=model_type,
                                             n_brightest_lines=cur_obj['n_brightest_lines'],
                                             turbulent_sigma=cur_obj['turbulent_sigma'],
                                             sys_velocity=cur_obj['sys_velocity'],
                                             linerat_constant=cur_obj['linerat_constant'],
                                             vel_gradient=cur_obj['vel_gradient'],
                                             vel_pa=cur_obj['vel_pa'],
                                             )

                elif cur_obj['type'] == "Filament":
                    generated_object = Filament(xc=x, yc=y,
                                                max_brightness=cur_obj.get('max_brightness'),
                                                max_extinction=cur_obj.get('max_extinction'),
                                                width=cur_obj['width'],
                                                length=cur_obj['length'],
                                                PA=cur_obj['PA'],
                                                vel_gradient=cur_obj['vel_gradient'],
                                                vel_pa=cur_obj['vel_pa'],
                                                spectrum_id=model_index,
                                                spectrum_type=model_type,
                                                n_brightest_lines=cur_obj['n_brightest_lines'],
                                                turbulent_sigma=cur_obj['turbulent_sigma'],
                                                sys_velocity=cur_obj['sys_velocity'],
                                                pxscale=self.pxscale * (cur_obj['distance'].to(u.pc) /
                                                                        self.distance.to(u.pc)),
                                                )

                elif cur_obj['type'] == "Galaxy":
                    generated_object = Galaxy(xc=x, yc=y,
                                              max_brightness=cur_obj.get('max_brightness'),
                                              max_extinction=cur_obj.get('max_extinction'),
                                              r_eff=cur_obj['r_eff'],
                                              rad_lim=cur_obj['rad_lim'],
                                              ax_ratio=cur_obj['ax_ratio'],
                                              PA=cur_obj['PA'],
                                              n=cur_obj['n'],
                                              vel_rot=cur_obj.get('vel_rot'),
                                              vel_pa=cur_obj['vel_pa'],
                                              spectrum_id=model_index,
                                              spectrum_type=model_type,
                                              n_brightest_lines=cur_obj['n_brightest_lines'],
                                              turbulent_sigma=cur_obj['turbulent_sigma'],
                                              sys_velocity=cur_obj['sys_velocity'],
                                              pxscale=self.pxscale * (cur_obj['distance'].to(u.pc) /
                                                                      self.distance.to(u.pc)),
                                              )
                elif cur_obj['type'] == "Ellipse":
                    generated_object = Ellipse(xc=x, yc=y,
                                               max_brightness=cur_obj.get('max_brightness'),
                                               max_extinction=cur_obj.get('max_extinction'),
                                               radius=cur_obj['radius'],
                                               ax_ratio=cur_obj['ax_ratio'],
                                               PA=cur_obj['PA'],
                                               spectrum_id=model_index,
                                               spectrum_type=model_type,
                                               n_brightest_lines=cur_obj['n_brightest_lines'],
                                               turbulent_sigma=cur_obj['turbulent_sigma'],
                                               sys_velocity=cur_obj['sys_velocity'],
                                               vel_gradient=cur_obj['vel_gradient'],
                                               vel_pa=cur_obj['vel_pa'],
                                               pxscale=self.pxscale * (cur_obj['distance'].to(u.pc) /
                                                                       self.distance.to(u.pc)),
                                               perturb_scale=cur_obj['perturb_scale'],
                                               perturb_amplitude=cur_obj['perturb_amplitude'],
                                               )
                elif cur_obj['type'] == "Circle":
                    generated_object = Circle(xc=x, yc=y,
                                              max_brightness=cur_obj.get('max_brightness'),
                                              max_extinction=cur_obj.get('max_extinction'),
                                              radius=cur_obj['radius'],
                                              spectrum_id=model_index,
                                              spectrum_type=model_type,
                                              n_brightest_lines=cur_obj['n_brightest_lines'],
                                              turbulent_sigma=cur_obj['turbulent_sigma'],
                                              sys_velocity=cur_obj['sys_velocity'],
                                              vel_gradient=cur_obj['vel_gradient'],
                                              vel_pa=cur_obj['vel_pa'],
                                              pxscale=self.pxscale * (cur_obj['distance'].to(u.pc) /
                                                                      self.distance.to(u.pc)),
                                              perturb_scale=cur_obj['perturb_scale'],
                                              perturb_amplitude=cur_obj['perturb_amplitude'],
                                              )
                elif cur_obj['type'] == "Rectangle" or (cur_obj['type'] == "Nebula"):
                    generated_object = Rectangle(xc=x, yc=y,
                                                 width=cur_obj.get('width'), height=cur_obj.get('height'),
                                                 max_brightness=cur_obj.get('max_brightness'),
                                                 max_extinction=cur_obj.get('max_extinction'),
                                                 spectrum_id=model_index,
                                                 spectrum_type=model_type,
                                                 n_brightest_lines=cur_obj['n_brightest_lines'],
                                                 turbulent_sigma=cur_obj['turbulent_sigma'],
                                                 sys_velocity=cur_obj['sys_velocity'],
                                                 vel_gradient=cur_obj['vel_gradient'],
                                                 vel_pa=cur_obj['vel_pa'],
                                                 pxscale=self.pxscale * (cur_obj['distance'].to(u.pc) /
                                                                         self.distance.to(u.pc)),
                                                 perturb_scale=cur_obj['perturb_scale'],
                                                 perturb_amplitude=cur_obj['perturb_amplitude'],
                                                 )
                elif cur_obj['type'] == "CustomNebula":
                    generated_object = self.process_custom_nebula(cur_obj, xc=x, yc=y,
                                                                  model_index=model_index)
                    if generated_object is None:
                        log.warning("Something wrong with the Custom Nebula: skip it")
                        continue
                else:
                    log.warning("Unrecognized type of the nebula #{0}: skip this one".format(ind_obj))
                    continue
            if model_index is not None:
                if cur_obj['linerat_constant']:
                    lr = "Constant"
                else:
                    lr = "Variable"
                add_fits_kw = {"Model_ID": model_id, "LineRat": lr}
            else:
                add_fits_kw = {}

            add_fits_kw["Distance"] = (cur_obj['distance'].to_value(u.kpc), "Distance to the nebula, kpc")

            continuum = None
            if cur_obj['continuum_type'] is not None and cur_obj['continuum_data'] is not None \
                    and cur_obj['continuum_type'].lower() in ['bb', 'poly', 'model']:

                if cur_obj['continuum_type'].lower() == 'model':
                    if isinstance(cur_obj['continuum_data'], dict) and ('wl' in cur_obj['continuum_data']) and \
                            ('flux' in cur_obj['continuum_data']):
                        if len(cur_obj['continuum_data']['wl']) != len(cur_obj['continuum_data']['flux']):
                            log.error("Number of wavelength and flux points is inconsistent for continuum")
                        else:
                            wlscale = cur_obj['continuum_data']['wl']
                            continuum = np.vstack([wlscale, cur_obj['continuum_data']['flux']])
                    elif ~isinstance(cur_obj['continuum_data'], dict) and lvmdatasimulator.CONTINUUM_MODELS is not None:
                        with fits.open(lvmdatasimulator.CONTINUUM_MODELS) as hdu:
                            if cur_obj['continuum_data'] >= hdu[0].data.shape[0]:
                                log.warning("Wrong continuum model ID for nebula #{0}".format(obj_id))
                            else:
                                wlscale = (np.arange(hdu[0].data.shape[1]) - hdu[0].header['CRPIX1'] + 1
                                           ) * hdu[0].header['CDELT1'] + hdu[0].header['CRVAL1']
                                continuum = np.vstack([wlscale, hdu[0].data[cur_obj['continuum_data']]])
                elif cur_obj['continuum_type'].lower() in ['poly', 'bb']:
                    continuum = cur_obj['continuum_data']
            if continuum is not None:
                if add_fits_kw is None:
                    add_fits_kw = {}
                add_fits_kw['CONTTYPE'] = (cur_obj['continuum_type'], "Type of the continuum")
                if cur_obj['continuum_flux'] > 0:
                    contflux = cur_obj['continuum_flux'].to_value(u.erg / u.cm ** 2 / u.s / u.arcsec ** 2 / u.AA)
                else:
                    contflux = 0
                add_fits_kw['CONTFLUX'] = (contflux,
                                           "Continuum brightness (in erg/s/cm^2/asec^2/AA) at ref. wl/Filter")
                if cur_obj['continuum_mag'] is not None:
                    contmag = cur_obj['continuum_mag'].to_value(u.mag/u.arcsec**2)
                else:
                    contmag = None
                add_fits_kw['CONTMAG'] = (contmag,
                                          "Continuum brightness (in mag/asec^2) at ref. wl/Filter")
                if isinstance(cur_obj['continuum_wl'], str):
                    cont_wl = cur_obj['continuum_wl']
                else:
                    cont_wl = cur_obj['continuum_wl'].to_value(u.AA)
                add_fits_kw['CONTWL'] = (cont_wl, 'Reference wavelength/filter for cont. flux/mag')

            if cur_obj.get('max_extinction') > 0:
                if add_fits_kw is None:
                    add_fits_kw = {}
                add_fits_kw['EXT_LAW'] = (cur_obj['ext_law'], "Extinction law according to pyneb list")
                add_fits_kw['EXT_RV'] = (cur_obj['ext_rv'], "R_V value for extinction calculations")

            self.add_nebula(generated_object, obj_id=obj_id, zorder=cur_obj.get('zorder'), add_fits_kw=add_fits_kw,
                            continuum=continuum)
            obj_id += 1
        bar.finish()
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
                if any(np.array([cur_hdu.header.get('EXTNAME') for cur_hdu in hdu
                                 if (cur_hdu.header.get('EXTNAME') is not None) and
                                    ("_LINEPROFILE" in cur_hdu.header.get('EXTNAME'))])):
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

    def calc_extinction(self, wavelength=6562.81, x0=0, y0=0, xs=None, ys=None, extension_name=None):
        """
        Calculate coefficient to reduce flux due to extinction at given wavelength(s)

        Args:
            x0: start x-coordinate in the field of view for calculations
            y0: start x-coordinate in the field of view for calculations
            xs: x-size (in pixels) of the area for calculations (if None => then just pixel x0,y0 is considered; xs=1)
            ys: y-size (in pixels) of the area for calculations (if None => then just pixel x0,y0 is considered; ys=1)
            wavelength: in angstrom, particular wavelength (or wavelengths)
                at which the calculations should be performed
            extension_name (str): name of the extension for current dark nebula

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

        if extension_name is None:
            all_dark_nebulae = [hdu.header.get('EXTNAME') for hdu in self.content if
                                hdu.header.get('EXTNAME') is not None and
                                ("BRIGHTNESS" in hdu.header.get('EXTNAME')) and
                                hdu.header.get('DARK') and
                                check_in_region(hdu.header.get('X0'), hdu.header.get('Y0'),
                                                hdu.header.get('NAXIS1'), hdu.header.get('NAXIS2'))]
        else:
            if not check_in_region(self.content[extension_name].header.get('X0'),
                                   self.content[extension_name].header.get('Y0'),
                                   self.content[extension_name].header.get('NAXIS1'),
                                   self.content[extension_name].header.get('NAXIS2')):
                return None
            all_dark_nebulae = [extension_name]
        if len(all_dark_nebulae) == 0:
            return None
        if type(wavelength) in [float, int, np.float64, np.float32]:
            wavelength = np.atleast_1d(wavelength)
        ext_map = np.ones(shape=(len(wavelength), y1 - y0 + 1, x1 - x0 + 1), dtype=np.float32)

        for dark_nebula in all_dark_nebulae:
            cur_neb_av = np.zeros(shape=(y1 - y0 + 1, x1 - x0 + 1), dtype=np.float32)
            cur_neb_x0 = np.clip(x0 - self.content[dark_nebula].header.get('X0'), 0, None)
            cur_neb_y0 = np.clip(y0 - self.content[dark_nebula].header.get('Y0'), 0, None)
            cur_neb_x1 = self.content[dark_nebula].header.get('NAXIS1') - 1 - np.clip(
                self.content[dark_nebula].header.get('X0') + self.content[dark_nebula].header.get('NAXIS1') - 1 - x1,
                0, None)
            cur_neb_y1 = self.content[dark_nebula].header.get('NAXIS2') - 1 - np.clip(
                self.content[dark_nebula].header.get('Y0') + self.content[dark_nebula].header.get('NAXIS2') - 1 - y1,
                0, None)
            cur_neb_av[cur_neb_y0 + self.content[dark_nebula].header.get('Y0') - y0:
                       cur_neb_y1 + self.content[dark_nebula].header.get('Y0') - y0 + 1,
                       cur_neb_x0 + self.content[dark_nebula].header.get('X0') - x0:
                       cur_neb_x1 + self.content[dark_nebula].header.get('X0') - x0 + 1
                       ] = self.content[dark_nebula].data[cur_neb_y0: cur_neb_y1 + 1, cur_neb_x0: cur_neb_x1 + 1]
            cur_extinction_law = self.ext_law
            cur_r_v = self.R_V
            if self.content[dark_nebula].header.get('EXT_LAW'):
                cur_extinction_law = self.content[dark_nebula].header.get('EXT_LAW')
            if self.content[dark_nebula].header.get('EXT_RV'):
                cur_r_v = self.content[dark_nebula].header.get('EXT_RV')
            ext_map = ext_map * ism_extinction(av=cur_neb_av, r_v=cur_r_v,
                                               ext_law=cur_extinction_law, wavelength=wavelength)
        return ext_map

    def get_map(self, wavelength=6562.81, get_continuum=False):
        """
        Method to produce 2D map of all ISM nebulae in the selected line
        Args:
            wavelength (float or iterative): wavelength (in Angstrom) according to the lines list, or wavelength range
            get_continuum (bool): if True, then also counts the flux from the continuum
        """
        wavelength = np.atleast_1d(wavelength)
        if len(wavelength) == 1:
            wavelength = np.array([wavelength[0]-0.01, wavelength[0]+0.01])
        if self.content[0].header['Nobj'] == 0:
            log.warning("ISM doesn't contain any nebula")
            return None
        all_extensions = [hdu.header.get('EXTNAME') for hdu in self.content]
        all_extensions_brt = np.array([extname for extname in all_extensions
                                       if extname is not None and ("BRIGHTNESS" in extname) and
                                       check_overlap(self.content[extname], (self.height, self.width))])

        if all([self.content[cur_ext].header.get("DARK") for cur_ext in all_extensions_brt]):
            # !!!! ADD later accounting of the continuum and extinction from those nebulae
            log.warning("ISM doesn't contain any emission nebula")
            return None

        all_extensions_brt = all_extensions_brt[
            np.argsort([self.content[cur_ext].header.get('ZORDER') for cur_ext in all_extensions_brt])]

        map_2d = np.zeros(shape=(self.height, self.width), dtype=np.float32)
        map_is_empty = True
        for cur_ext in all_extensions_brt:
            my_comp = "_".join(cur_ext.split("_")[:2])

            y0_in_field = np.clip(self.content[cur_ext].header['Y0'], 0, None)
            y1_in_field = np.clip(self.content[cur_ext].header['Y0'] + self.content[cur_ext].header['NAXIS2'] - 1, None,
                                  self.height-1)
            x0_in_field = np.clip(self.content[cur_ext].header['X0'], 0, None)
            x1_in_field = np.clip(self.content[cur_ext].header['X0'] + self.content[cur_ext].header['NAXIS1'] - 1, None,
                                  self.width - 1)

            if get_continuum and (my_comp + "_CONTINUUM" in all_extensions):
                n_wl_bins = int(np.clip(np.ceil((wavelength[1]-wavelength[0])/20.), 10, 200))
                wl_grid = np.linspace(wavelength[0], wavelength[1], n_wl_bins)
                continuum = np.sum(self._get_continuum(my_comp, wl_grid))
                add_continuum = self.content[cur_ext].data / np.max(self.content[cur_ext].data) * continuum
                map_2d[y0_in_field: y1_in_field + 1, x0_in_field: x1_in_field + 1] += \
                    add_continuum[y0_in_field - self.content[cur_ext].header['Y0']:
                                  y1_in_field - self.content[cur_ext].header['Y0'] + 1,
                                  x0_in_field - self.content[cur_ext].header['X0']: x1_in_field -
                                  self.content[cur_ext].header['X0'] + 1
                                  ]
                map_is_empty = False
            if self.content[cur_ext].header.get("DARK"):
                if map_is_empty:
                    continue
                ext_map = self.calc_extinction(wavelength=(wavelength[-1] + wavelength[0])/2., xs=self.width,
                                               ys=self.height,
                                               extension_name=cur_ext)
                if ext_map is not None:
                    map_2d = map_2d * ext_map[0]
                continue

            all_flux_wl = [extname[-7:].strip('_') for extname in all_extensions
                           if extname is not None and (my_comp in extname and "FLUX_" in extname)]
            all_flux_wl = np.array(all_flux_wl)

            # flux_ext = [extname for extname in all_extensions
            #             if extname is not None and (my_comp in extname and
            #                                         "FLUX_{0}".format(np.round(wavelength, 2)) in extname)]
            add_emission = np.zeros(shape=(self.content[cur_ext].header['NAXIS2'],
                                           self.content[cur_ext].header['NAXIS1']), dtype=np.float32)
            if len(all_flux_wl) == 0:
                fluxrat_ext = [extname for extname in all_extensions
                               if extname is not None and (my_comp in extname and "FLUXRATIOS" in extname)]
                if len(fluxrat_ext) == 0:
                    continue
                fluxrat_ext = fluxrat_ext[0]

                wl_indexes = np.flatnonzero((self.content[fluxrat_ext].data[0, :] > (wavelength[0])) &
                                            (self.content[fluxrat_ext].data[0, :] < (wavelength[1])))

                # wl_index = np.flatnonzero(np.isclose(self.content[fluxrat_ext].data[0, :], wavelength))
                if len(wl_indexes) == 0:
                    continue
                for wl_index in wl_indexes:
                    add_emission += (self.content[cur_ext].data * self.content[fluxrat_ext].data[1, wl_index])
            else:
                all_flux_wl_float = np.array(all_flux_wl).astype(np.float32)
                wl_indexes = np.flatnonzero((all_flux_wl_float > (wavelength[0] - 0.01)) &
                                            (all_flux_wl_float < (wavelength[1] + 0.01)))
                flux_ext_wl = all_flux_wl[wl_indexes]
                if len(wl_indexes) == 0:
                    continue

                for cur_wl in flux_ext_wl:
                    add_emission += self.content[my_comp + "_FLUX_" + cur_wl].data

            map_2d[y0_in_field: y1_in_field + 1, x0_in_field: x1_in_field + 1] += \
                add_emission[y0_in_field - self.content[cur_ext].header['Y0']:
                             y1_in_field - self.content[cur_ext].header['Y0'] + 1,
                             x0_in_field - self.content[cur_ext].header['X0']: x1_in_field -
                             self.content[cur_ext].header['X0'] + 1
                             ]

            map_is_empty = False
        return map_2d * (proj_plane_pixel_scales(self.wcs)[0] * 3600) ** 2

    def get_spectrum(self, wl_grid=None, aperture_mask=None, fibers_coords=None, pix_size=1):
        if aperture_mask is None or (np.sum(aperture_mask) == 0) or (self.content[0].header['Nobj'] == 0):
            log.warning("No overlapping detected between the ISM component and the fibers => no spectra extraction")
            return None
        all_extensions = [hdu.header.get('EXTNAME') for hdu in self.content]
        all_extensions_brt = np.array([extname for extname in all_extensions
                                       if extname is not None and ("BRIGHTNESS" in extname) and
                                       check_overlap(self.content[extname], (self.height, self.width))])
        if all([self.content[cur_ext].header.get("DARK") for cur_ext in all_extensions_brt]):
            return None
        all_extensions_brt = all_extensions_brt[np.argsort([self.content[cur_ext].header.get('ZORDER')
                                                            for cur_ext in all_extensions_brt])]

        wl_logscale = np.log(wl_grid.value)
        wl_logscale_highres = np.arange((np.round(wl_logscale[-1] - wl_logscale[0]) * 1e6
                                         ).astype(int)) * 1e-6 + np.round(wl_logscale[0], 6)
        delta_lr = np.roll(wl_logscale, -1) - wl_logscale
        delta_lr[-1] = delta_lr[-2]

        xx, yy = np.meshgrid(np.arange(aperture_mask.shape[1]), np.arange(aperture_mask.shape[0]))
        pix_in_apertures = aperture_mask > 0
        xstart = np.clip(np.min(xx[pix_in_apertures]) - 1, 0, None)
        ystart = np.clip(np.min(yy[pix_in_apertures]) - 1, 0, None)
        xfin = np.clip(np.max(xx[pix_in_apertures]) + 1, None, aperture_mask.shape[1]-1)
        yfin = np.clip(np.max(yy[pix_in_apertures]) + 1, None, aperture_mask.shape[0]-1)
        aperture_mask_sub = aperture_mask[ystart: yfin + 1, xstart: xfin + 1]
        xx_sub, yy_sub = np.meshgrid(np.arange(xfin - xstart + 1), np.arange(yfin - ystart + 1))
        n_apertures = np.max(aperture_mask)
        # aperture_centers = np.round(fibers_coords).astype(int)
        spectrum = np.zeros(shape=(n_apertures, len(wl_grid)), dtype=np.float32)

        fiber_radius = np.median(fibers_coords[:, 2])
        kern_mask = calc_circular_mask(fiber_radius)
        kern = kernels.CustomKernel(kern_mask.reshape((1, kern_mask.shape[0], kern_mask.shape[1])))

        bar = progressbar.ProgressBar(max_value=len(all_extensions_brt)).start()
        for neb_index, cur_ext in enumerate(all_extensions_brt):
            cur_neb_in_mask = np.zeros_like(xx_sub)
            y0 = self.content[cur_ext].header.get("Y0")
            x0 = self.content[cur_ext].header.get("X0")
            nx = self.content[cur_ext].header.get('NAXIS1')
            ny = self.content[cur_ext].header.get('NAXIS2')
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
            selected_apertures = np.flatnonzero(((fibers_coords[:, 0] + fibers_coords[:, 2]) >= x0) &
                                                ((fibers_coords[:, 0] - fibers_coords[:, 2]) <= (x0 + nx - 1)) &
                                                ((fibers_coords[:, 1] + fibers_coords[:, 2]) >= y0) &
                                                ((fibers_coords[:, 1] - fibers_coords[:, 2]) <= (y0 + ny - 1)))
            selected_apertures = np.array([sa for sa in selected_apertures if (sa+1) in aperture_mask_sub], dtype=int)
            if len(selected_apertures) == 0:
                bar.update(neb_index + 1)
                continue

            # Here I check if it is necessary to extend all involved arrays to account for fibers at the edges of neb.
            dx0 = abs(np.clip(np.round(np.min(fibers_coords[selected_apertures, 0] -
                                              fibers_coords[selected_apertures, 2])).astype(int) - 1 - x0, None, 0))
            dx1 = np.clip(np.round(np.max(fibers_coords[selected_apertures, 0] +
                                          fibers_coords[selected_apertures, 2])).astype(int) + 2 - x0 - nx, 0, None)
            dy0 = abs(np.clip(np.round(np.min(fibers_coords[selected_apertures, 1] -
                                              fibers_coords[selected_apertures, 2])).astype(int) - 1 - y0, None, 0))
            dy1 = np.clip(np.round(np.max(fibers_coords[selected_apertures, 1] +
                                          fibers_coords[selected_apertures, 2])).astype(int) + 2 - y0 - ny, 0, None)
            if (dx0 > 0) or (dy0 > 0) or (dx1 > 0) or (dy1 > 0):
                npad = ((0, 0), (dy0, dy1), (dx0, dx1))
            else:
                npad = None

            selected_y = fibers_coords[selected_apertures, 1] - ystart_neb - y0 + dy0
            selected_x = fibers_coords[selected_apertures, 0] - xstart_neb - x0 + dx0

            if self.content[cur_ext].header.get("DARK"):
                extinction_map = self.content[cur_ext].data[cur_mask_in_neb].reshape((1, yfin_neb - ystart_neb + 1,
                                                                                      xfin_neb - xstart_neb + 1))
                if npad is not None:
                    extinction_map = np.pad(extinction_map, pad_width=npad, mode='constant', constant_values=0)

                if self.content[cur_ext].header.get('EXT_LAW'):
                    cur_extinction_law = self.content[cur_ext].header.get('EXT_LAW')
                else:
                    cur_extinction_law = self.ext_law
                if self.content[cur_ext].header.get('EXT_RV'):
                    cur_r_v = self.content[cur_ext].header.get('EXT_RV')
                else:
                    cur_r_v = self.R_V

                data_in_apertures = convolve_array(extinction_map,
                                                   kern, selected_x, selected_y, pix_size,
                                                   nchunks=self.content[cur_ext].header.get("NCHUNKS"),
                                                   normalize_kernel=True)

                # data_in_apertures = convolve_fft(extinction_map, kern,
                #                                  allow_huge=True, normalize_kernel=True)[
                #                     :, aperture_centers[selected_apertures, 1] - ystart_neb - y0 + dy0,
                #                     aperture_centers[selected_apertures, 0] - xstart_neb - x0 + dx0]
                data_in_apertures = data_in_apertures.reshape((data_in_apertures.shape[0] * data_in_apertures.shape[1],
                                                               1))
                spectrum[selected_apertures, :] = \
                    spectrum[selected_apertures, :] * ism_extinction(av=data_in_apertures, r_v=cur_r_v,
                                                                     ext_law=cur_extinction_law, wavelength=wl_grid).T

                bar.update(neb_index + 1)
                continue

            my_comp = "_".join(cur_ext.split("_")[:2])

            if self.content[cur_ext].header.get('MAXBRT'):
                if self.content[cur_ext].header.get("LINERAT") == 'Variable':
                    all_wavelength = np.array([extname.split("_")[-1] for extname in all_extensions
                                               if extname is not None and (my_comp + "_FLUX_" in extname)])
                else:
                    all_wavelength = self.content[my_comp + "_FLUXRATIOS"].data[0, :]

                if self.content[cur_ext].header.get("LINERAT") == 'Variable':
                    all_fluxes = np.array([self.content[my_comp + "_FLUX_" + wl].data[
                                               cur_mask_in_neb].reshape((yfin_neb - ystart_neb + 1,
                                                                         xfin_neb - xstart_neb + 1))
                                           for wl in all_wavelength], dtype=np.float32)
                    all_wavelength = all_wavelength.astype(np.float32)
                else:
                    all_fluxes = self.content[cur_ext].data[cur_mask_in_neb].reshape((1, yfin_neb - ystart_neb + 1,
                                                                                      xfin_neb - xstart_neb + 1)
                                                                                     ).astype(np.float32)

                if my_comp + "_LINEPROFILE" in self.content:
                    lsf = self.content[my_comp + "_LINEPROFILE"].data[
                          :, cur_mask_in_neb].reshape((len(self.vel_grid), yfin_neb - ystart_neb + 1,
                                                       xfin_neb - xstart_neb + 1)).astype(np.float32)
                else:
                    if my_comp + "_VEL" in self.content:
                        vel = self.content[my_comp + "_VEL"].data[cur_mask_in_neb].reshape((yfin_neb - ystart_neb + 1,
                                                                                            xfin_neb - xstart_neb + 1))
                    else:
                        vel = np.zeros(shape=(yfin_neb - ystart_neb + 1, xfin_neb - xstart_neb + 1),
                                       dtype=np.float32)  # + self.content[cur_ext].header.get('SysVel')
                        # self.sys_velocity.value
                    if my_comp + "_DISP" in self.content:
                        disp = self.content[my_comp + "_DISP"].data[
                            cur_mask_in_neb].reshape((yfin_neb - ystart_neb + 1, xfin_neb - xstart_neb + 1))
                    else:
                        disp = np.zeros(shape=(yfin_neb - ystart_neb + 1, xfin_neb - xstart_neb + 1),
                                        dtype=np.float32) + self.content[cur_ext].header.get('TurbVel')
                        # self.turbulent_sigma.value
                    lsf = np.exp(-np.power(
                        (self.vel_grid.value[:, None, None] - vel[None, :, :]) / disp[None, :, :], 2.) / 2)
                    lsf = (lsf / np.nansum(lsf, axis=0)).astype(np.float32)

                if npad is not None:
                    lsf = np.pad(lsf, pad_width=npad, mode='constant', constant_values=0)
                    all_fluxes = np.pad(all_fluxes, pad_width=npad, mode='constant', constant_values=0)

                data_in_apertures = [convolve_array(lsf * line_data[None, :, :],
                                                    kern, selected_x, selected_y, pix_size,
                                                    nchunks=self.content[cur_ext].header.get("NCHUNKS"))
                                     for line_data in all_fluxes]
                data_in_apertures = np.array(data_in_apertures)
                if data_in_apertures.shape == 2:
                    data_in_apertures = data_in_apertures.reshape((1, data_in_apertures.shape[0],
                                                                   data_in_apertures.shape[1]))

                # if all_fluxes.shape[0] == 1:
                #     # selected_y = fibers_coords[selected_apertures, 1] - ystart_neb - y0 + dy0
                #     # selected_x = fibers_coords[selected_apertures, 0] - xstart_neb - x0 + dx0
                #     data_in_apertures = convolve_array(lsf * all_fluxes[0][None, :, :],
                #                                        kern, selected_x, selected_y, pix_size,
                #                                        nchunks=self.content[cur_ext].header.get("NCHUNKS"))
                #     data_in_apertures = data_in_apertures.reshape((1, data_in_apertures.shape[0],
                #                                                   data_in_apertures.shape[1]))
                #     # hdu = fits.PrimaryHDU(data=data_in_apertures)
                #     # hdu.writeto(f'./data_{neb_index}.fits', overwrite=True)
                #
                # else:
                #
                #     # selected_y = fibers_coords[selected_apertures, 1] - ystart_neb - y0 + dy0
                #     # selected_x = fibers_coords[selected_apertures, 0] - xstart_neb - x0 + dx0
                #     # data_in_apertures = Parallel(n_jobs=lvmdatasimulator.n_process)(
                #     #     delayed(convolve_array)(lsf * line_data[None, :, :], kern,
                #     #                            selected_y, selected_x, pix_size)
                #     #     for line_data in all_fluxes)
                #     data_in_apertures = [convolve_array(lsf * line_data[None, :, :], kern,
                #                                         selected_y, selected_x, pix_size,
                #                                         nchunks=self.content[cur_ext].header.get("NCHUNKS"))
                #                          for line_data in all_fluxes]
                #     data_in_apertures = np.array(data_in_apertures)
                #     # hdu = fits.PrimaryHDU(data=data_in_apertures)
                #     # hdu.writeto(f'./data_{neb_index}.fits', overwrite=True)

                data_in_apertures = np.moveaxis(data_in_apertures, 2, 0)
                if data_in_apertures.shape[1] > 1:
                    prf_index = np.flatnonzero(all_wavelength == 6562.81)
                    if len(prf_index) == 0:
                        prf_index = np.flatnonzero(np.round(all_wavelength) == 6563.)
                        if len(prf_index) == 0:
                            prf_index = 0
                else:
                    prf_index = 0
                flux_norm_in_apertures = np.nansum(data_in_apertures, axis=2)
                line_prf_in_apertures = data_in_apertures[:, prf_index, :].reshape(
                    (data_in_apertures.shape[0], data_in_apertures.shape[2])) / \
                    flux_norm_in_apertures[:, prf_index].reshape(data_in_apertures.shape[0], 1)

                wl_logscale_lsf = np.log((self.vel_grid.value + self.content[cur_ext].header.get('SysVel')
                                          ) / 2.9979e5 + 1)
                highres_factor = 1e6
                wl_logscale_lsf_highres = np.arange(np.round((wl_logscale_lsf[-1] - wl_logscale_lsf[0]
                                                              ) / 2. * highres_factor).astype(int) * 2 + 1
                                                    ) / highres_factor + wl_logscale_lsf[0]
                p = interp1d(wl_logscale_lsf, line_prf_in_apertures, axis=1, assume_sorted=True,
                             fill_value=0, bounds_error=False)
                line_highres_log = p(wl_logscale_lsf_highres)
                line_highres_log = line_highres_log / np.sum(line_highres_log, axis=1)[:, None]
                if flux_norm_in_apertures.shape[1] == 1:
                    flux_norm_in_apertures = flux_norm_in_apertures * \
                                        self.content[my_comp + "_FLUXRATIOS"].data[1, None, :]
                wl_indexes = np.round((np.log(all_wavelength * (1 + self.content[cur_ext].header.get('SysVel') /
                                                                2.9979e5)) - wl_logscale_highres[0]
                                       ) * highres_factor).astype(int)
                rec = (wl_indexes > 0) & (wl_indexes < len(wl_logscale_highres))
                spectrum_highres_log = np.zeros(shape=(len(selected_apertures), len(wl_logscale_highres)),
                                                dtype=np.float32)
                win = (len(wl_logscale_lsf_highres) - 1) // 2
                for ind, r in enumerate(rec):
                    if r:
                        spectrum_highres_log[:, wl_indexes[ind] - win: wl_indexes[ind] + win + 1] += \
                            line_highres_log[:, :] * flux_norm_in_apertures[:, ind].reshape(
                                (len(selected_apertures), 1))
                p = interp1d(wl_logscale_highres, spectrum_highres_log, axis=1, assume_sorted=True, bounds_error=False,
                             fill_value='extrapolate')
                spectrum[selected_apertures, :] += (p(wl_logscale) * delta_lr * highres_factor)

            if my_comp + "_CONTINUUM" in self.content:
                brt_max = self.content[cur_ext].header.get('MAXBRT')
                if not brt_max:
                    brt_max = self.content[cur_ext].header.get('MAXEXT')
                if not brt_max:
                    brt_max = 1
                continuum = self._get_continuum(my_comp, wl_grid.value)

                brt = self.content[cur_ext].data[cur_mask_in_neb].reshape((1, yfin_neb - ystart_neb + 1,
                                                                           xfin_neb - xstart_neb + 1)) / brt_max
                if brt.shape[0] != 1:
                    brt = brt.reshape((1, brt.shape[0], brt.shape[1]))
                if npad is not None:
                    brt = np.pad(brt, pad_width=npad, mode='constant', constant_values=0)
                # selected_y = fibers_coords[selected_apertures, 1] - ystart_neb - y0 + dy0
                # selected_x = fibers_coords[selected_apertures, 0] - xstart_neb - x0 + dx0
                data_in_apertures = convolve_array(brt,
                                                   kern, selected_x, selected_y, pix_size,
                                                   nchunks=self.content[cur_ext].header.get("NCHUNKS"))
                # data_in_apertures = convolve_fft(brt, kern,
                #                                  allow_huge=True, normalize_kernel=False)[
                #                     :, aperture_centers[selected_apertures, 1] - ystart_neb - y0 + dy0,
                #                     aperture_centers[selected_apertures, 0] - xstart_neb - x0 + dx0]
                data_in_apertures = data_in_apertures.reshape((data_in_apertures.shape[0] * data_in_apertures.shape[1],
                                                               1))
                spectrum[selected_apertures, :] += continuum[None, :] * data_in_apertures

            bar.update(neb_index + 1)
        bar.finish()
        return spectrum * (proj_plane_pixel_scales(self.wcs)[0] * 3600) ** 2 * fluxunit * u.arcsec ** 2
