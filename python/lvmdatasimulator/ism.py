# encoding: utf-8
#
# @Author: Oleg Egorov, Enrico Congiu
# @Date: Nov 15, 2021
# @Filename: ism.py
# @License: BSD 3-Clause
# @Copyright: Oleg Egorov, Enrico Congiu

# import scipy.optimize
from astropy import units as u
from astropy import constants as c
import numpy as np
from matplotlib import pyplot as plt
# from scipy.integrate import nquad
# import tqdm
# from multiprocessing import Pool
# import multiprocessing
# from scipy.special import sph_harm
from astropy.wcs import WCS
from astropy.table import QTable
from dataclasses import dataclass,field
import functools
from scipy.ndimage.interpolation import map_coordinates
from scipy.interpolate import interp1d
from typing import List
from lvmdatasimulator import log

fluxunit = u.erg / (u.cm ** 2 * u.s * u.arcsec ** 2)
velunit = u.km / u.s


@dataclass
class Nebula:
    """
    Base class defining properties of every nebula type.
    By itself it describes the rectangular nebula (for now - as DIG)
    """
    xc: int = 0  # Center of the region in the field of view, pix
    yc: int = 0  # Center of the region in the field of view, pix
    width: int = 101  # full width of cartesian grid, pix (should be odd)
    height: int = 101  # full height of cartesian grid, pix (should be odd)
    pxscale: u.pc = 0.01 * u.pc  # pixel size in pc
    spectrum_id: int = 0  # ID of a template Cloudy spectrum for this nebula
    sys_velocity: velunit = 0 * velunit  # Systemic velocity
    max_brightness: fluxunit = 1e-15 * fluxunit
    max_extinction: u.mag = 0 * u.mag
    _npix_los: int = 1  # full size along line of sight in pixels

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
    _theta_bins: int = 20
    _rad_bins: int = 10
    _h_bins: int = 2
    _npix_los: int = 51

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

        theta_c[theta_c < (0 * u.radian)] = theta_c[theta_c < (0 * u.radian)] + 2 * np.pi * u.radian
        theta_c[theta_c > (2 * np.pi * u.radian)] = -2 * np.pi * u.radian + theta_c[theta_c > (2 * np.pi * u.radian)]

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


@dataclass
class Cloud(Nebula):
    """Class of an isotropic spherical gas cloud without any ionization source.
    Defined by its position, radius, density, maximal optical depth"""
    radius: u.pc = 1.0 * u.pc
    max_brightness: fluxunit = 0 * fluxunit
    max_extinction: u.mag = 2.0 * u.mag
    thickness: float = 1.0
    _phi_bins: int = 180
    _theta_bins: int = 180
    _rad_bins: int = 100
    _npix_los: int = 100

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
            3D cube of normalized brightness in theta-phi-rad grid; total brightness = 1 (0 in case of thickness = 0)
        """
        rho, theta, phi = np.meshgrid(self._rad_grid, self._theta_grid, self._phi_grid, indexing='ij')
        brt = np.ones_like(theta)
        brt[rho < (self.radius * (1 - self.thickness))] = 0
        brt[rho > self.radius] = 0
        norm = np.sum(brt)
        if norm > 0:
            brt = brt / np.sum(brt)
        return brt

    @functools.cached_property
    def _brightness_3d_cartesian(self):
        x, y, z = np.meshgrid(self._cartesian_x_grid,
                              self._cartesian_y_grid,
                              self._cartesian_z_grid, indexing='ij')

        phi_c = (np.arctan2(y, x))
        rad_c = (np.sqrt(x ** 2 + y ** 2 + z ** 2))
        rad_c[rad_c == 0 * u.pc] = 1e-3 * self.pxscale
        theta_c = (np.arccos(z / rad_c))

        phi_c[phi_c < (0 * u.radian)] = 2 * np.pi * u.radian + phi_c[phi_c < (0 * u.radian)]
        phi_c[phi_c > (2 * np.pi * u.radian)] = -2 * np.pi * u.radian + phi_c[phi_c >
                                                                              (2 * np.pi * u.radian)]
        theta_c[theta_c < (0 * u.radian)] = np.pi * u.radian + theta_c[theta_c < (0 * u.radian)]
        theta_c[theta_c > (np.pi * u.radian)] = -np.pi * u.radian + theta_c[theta_c > (np.pi * u.radian)]

        ir = interp1d(self._rad_grid, np.arange(self._rad_bins), bounds_error=False)
        ith = interp1d(self._theta_grid, np.arange(self._theta_bins))
        iphi = interp1d(self._phi_grid, np.arange(self._phi_bins))
        new_ir = ir(rad_c.ravel())
        new_ith = ith(theta_c.ravel())
        new_iphi = iphi(phi_c.ravel())
        cart_data = map_coordinates(self._brightness_3d_spherical,
                                    np.vstack([new_ir, new_ith, new_iphi]),
                                    order=1, mode='constant', cval=0)
        return cart_data.reshape([len(self._cartesian_x_grid),
                                  len(self._cartesian_y_grid),
                                  len(self._cartesian_z_grid)]).T


@dataclass
class Bubble(Cloud):
    """Class of an isotropic thin expanding bubble."""
    spectral_axis: velunit = np.arange(-20,20,10) * velunit
    expansion_velocity: velunit = 20 * velunit
    turbulent_sigma: velunit = 30 * velunit
    max_brightness: fluxunit = 1e-15 * fluxunit
    max_extinction: u.mag = 0 * u.mag
    thickness: float = 0.2
    quad_epsrel = 1e-2

    def velocity(self, theta: float, phi: float) -> u.km / u.s:
        """
        Method to calculate expansion velocity at given theta, phi

        Args:
            theta: float -- polar angle [0, np.pi]
            phi: float -- azimuthal angle [0, 2 * np.pi]

        Returns:
            Expansion velocity, in astropy.Quantity
        """
        return self.expansion_velocity * np.cos(theta)

    def d_area(self, theta: float, rad:u.pc):
        """Area differential dS/dφdθ, in unitless pc**2"""
        return rad.to_value(u.pc) ** 2 * np.sin(theta)

    # def area(self):
    #     return nquad(self.d_area, [[0, np.pi], [0, 2 * np.pi]])

    def turbulent_lsf(self, theta, phi, velocity):
        """Line spread function as a function of coorinates, including the velocity center shift"""
        mu = self.velocity(theta, phi)
        sig = self.turbulent_sigma
        return 1. / (np.sqrt(2. * np.pi) * sig) * np.exp(-np.power((velocity - mu) / sig, 2.) / 2)

    def d_spectrum(self, theta: float, phi: float, rad:u.pc, velocity: u.km / u.s):
        """Returns local spectrum, per pc**2 of area"""
        return (
                self._brightness_3d_spherical[:, :, :, None] * (fluxunit / u.pc **2)
                * self.d_area(theta, rad) * u.pc ** 2
                * self.turbulent_lsf(theta, phi, velocity)
        ).to(fluxunit / velunit)

    def spectrum(self) -> (fluxunit / velunit):
        velocities = self.spectral_axis.to(velunit, equivalencies=u.spectral())
        rad, theta, phi, velocities = np.meshgrid(self._rad_grid, self._theta_grid, self._phi_grid,
                                                  velocities, indexing='ij')
        spectrum = (
                np.sum(self.d_spectrum(theta, phi, rad, velocities), axis=(0, 1, 2))
                * (self._theta_grid[1] - self._theta_grid[0])
                * (self._phi_grid[1] - self._phi_grid[0])
                * (self._rad_grid[1] - self._rad_grid[0])
        )
        return spectrum


#
# @dataclass
# class MultipoleBubble(Bubble):
#     """Same as `Bubble`, but multipole brightness"""
#     maxdegree: int = 4
#     harm_amplitudes: np.ndarray = np.ones(maxdegree * (maxdegree + 2))
#
#     def brightness(self, theta: float, phi: float):
#         brightness = super().brightness(theta, phi)
#         if self.maxdegree > 0:
#             if len(self.harm_amplitudes) != (self.maxdegree * (self.maxdegree + 2)):
#                 self.harm_amplitudes = np.ones(self.maxdegree * (self.maxdegree + 2))
#             i = 0
#             for l in np.arange(1, self.maxdegree+1):
#                 for m in np.arange(-l, l+1):
#                     brightness += (self.harm_amplitudes[i] * (sph_harm(m, l, phi, theta).real) * self.brightness_norm)
#                     i += 1
#         return brightness
#
#
#
# @dataclass
# class MultipoleBubbleAffectingVelocity(MultipoleBubble):
#     def velocity(self, theta: float, phi: float) -> u.km / u.s:
#         velocity = super().velocity(theta, phi) / self.brightness(theta, phi) * self.brightness_norm
#         return velocity


@dataclass
class ISM:
    """
    Class defining ISM contribution to the field of view
    """
    wcs: WCS
    width: int = 400  # Width of field of view in pixels
    height: int = 400  # Width of field of view in pixels
    spec_resolution: u.Angstrom = 0.06 * u.Angstrom  # Spectral resolution of the simulation
    npix_line: int = 1  # Minimal number of pixels for a resolution element at wl = 10000A for construction of vel.grid
    distance: u.kpc = 50 * u.kpc  # Distance to the object for further conversion between arcsec and pc
    sys_velocity: velunit = 0 * velunit  # Systemic velocity to center the vel.grid on
    vel_amplitude : velunit = 100 * velunit  # Maximal deviation from the systemic velocity to setup vel.grid
    nebulae: List = List[None]
    turb_velocity: velunit = 15. * velunit
    last_id: int = 0
    ext_eps: u.mag = 0.01 * u.mag
    brt_eps: fluxunit = 1e-20 * fluxunit
    vel_contrib_eps: float = 1e-3

    def __post_init__(self):
        self.content = QTable(names=["ObjID", 'X', 'Y', 'Brightness', 'Extinction',
                                     'LineProfile', 'TurbVelocity', 'SpectrumID', 'Zorder'],
                              dtype=(int, int, int, float, float, np.ndarray, float, int, int),
                              units={'Brightness': fluxunit, 'Extinction': u.mag, 'TurbVelocity': velunit})
        self.vel_grid = np.linspace(-self.vel_amplitude + self.sys_velocity,
                                    self.vel_amplitude + self.sys_velocity,
                                    np.ceil(self.vel_amplitude / self.vel_resolution).astype(int) * 2 + 1)

    @functools.cached_property
    def vel_resolution(self):
        return (self.spec_resolution / self.npix_line / (10000 * u.Angstrom) * c.c).to(velunit)

    def _add_pix_to_tab(self, params):
        x = params[0]
        y = params[1]
        obj_to_add = self._cur_obj_to_add
        turb_velocity = params[2]
        zorder = params[3]
        spectrum_id = params[4]

        if obj_to_add.max_brightness > 0:
            cur_brt = obj_to_add.brightness_skyplane[x, y]
            if cur_brt < self.brt_eps:
                cur_brt = 0
        else:
            cur_brt = 0
        if obj_to_add.max_extinction > 0:
            cur_ext = obj_to_add.extinction[x, y]
            if cur_ext < self.ext_eps:
                cur_ext = 0
        else:
            cur_ext = 0

        if cur_brt == 0 and cur_ext == 0:
            return False

        if len(obj_to_add.los_velocity) <= 3:
            cur_lineprofile = np.zeros(shape=len(self.vel_grid), dtype=float)
            idx = (np.abs(self.vel_grid - obj_to_add.sys_velocity)).argmin()
            cur_lineprofile[idx] = 1.
        else:
            ip = interp1d(obj_to_add.los_velocity, obj_to_add.line_profile, fill_value=0.,
                          assume_sorted=True, bounds_error=False, copy=False)
            cur_lineprofile = ip(self.vel_grid)
            cur_lineprofile[cur_lineprofile < self.vel_contrib_eps] = 0
        dx = (len(obj_to_add._cartesian_x_grid) - 1) / 2
        dy = (len(obj_to_add._cartesian_y_grid) - 1) / 2
        self.content.add_row([self.last_id, x - dx + obj_to_add.xc, y - dy + obj_to_add.yc, cur_brt, cur_ext,
                              cur_lineprofile, turb_velocity, spectrum_id, zorder])  # , obj_to_add])
        return True

    def add_nebula(self, obj_to_add, spectrum_id=0, zorder=0, turb_velocity=turb_velocity):
        """
        Method to add the particular nebula to the ISM object and to the output
        """
        if type(obj_to_add) not in [Nebula, Bubble, Filament, DIG, Cloud]:
            log.warning('Skip nebula of wrong type ({})'.format(type(obj_to_add)))
            return

        if (obj_to_add.max_brightness <= 0) and (obj_to_add.max_extinction <= 0):
            log.warning('Skip nebula with zero extinction and brightness')
            return

        self._cur_obj_to_add = obj_to_add
        ny = len(obj_to_add._cartesian_y_grid)
        nx = len(obj_to_add._cartesian_x_grid)
        npix = nx * ny
        xx, yy = np.meshgrid(np.arange(nx),np.arange(ny))
        params = zip(xx.ravel(), yy.ravel(),
                     [turb_velocity] * npix, [zorder] * npix, [spectrum_id] * npix)

        # for p in params:
        #     self._add_pix_to_tab(p)
        # with Pool() as p:
        #     r = p.map(self._add_pix_to_tab, params)

        #     _ = list(tqdm.tqdm(p.imap(self._add_pix_to_tab, params), total=npix))

        for xy in zip(xx.ravel(),yy.ravel()):
            if obj_to_add.max_brightness > 0:
                cur_brt = obj_to_add.brightness_skyplane[xy[0],xy[1]]
                if cur_brt < self.brt_eps:
                    cur_brt = 0
            else:
                cur_brt = 0
            if obj_to_add.max_extinction > 0:
                cur_ext = obj_to_add.extinction[xy[0], xy[1]]
                if cur_ext < self.ext_eps:
                    cur_ext = 0
            else:
                cur_ext = 0

            if cur_brt == 0 and cur_ext == 0:
                continue

            if len(obj_to_add.los_velocity) <= 3:
                cur_lineprofile = np.zeros(shape=len(self.vel_grid), dtype=float)
                idx = (np.abs(self.vel_grid - obj_to_add.sys_velocity)).argmin()
                cur_lineprofile[idx] = 1.
            else:
                ip = interp1d(obj_to_add.los_velocity, obj_to_add.line_profile, fill_value=0.,
                              assume_sorted=True, bounds_error=False, copy=False)
                cur_lineprofile = ip(self.vel_grid)
                cur_lineprofile[cur_lineprofile < self.vel_contrib_eps] = 0

            self.content.add_row([self.last_id, xy[0], xy[1], cur_brt, cur_ext, cur_lineprofile,
                                  turb_velocity, spectrum_id, zorder])#, obj_to_add])
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
                                expansion_velocity: 10 * u.km/s,
                                }]
        """
        if type(all_objects) not in [list, tuple]:
            log.warning('Cannot generate nebulae as the input is not a list or tuple')
            return
        all_objects = [cobj for cobj in all_objects if cobj.get('type') in ['Nebula', 'Bubble',
                                                'Filament', 'DIG', 'Cloud']]
        n_objects = len(all_objects)
        log.info("Start generating {} nebulae".format(n_objects))

        for cur_obj in all_objects:
            if not (cur_obj.get('max_brightness') or cur_obj.get('max_extinction')):
                continue
            if cur_obj['type'] == 'DIG':
                generated_object = DIG()
            else:
                if not ((cur_obj.get('RA') and cur_obj.get('DEC')) or
                        (cur_obj.get('X') and cur_obj.get('Y'))):
                    continue
                if not (cur_obj.get('X') and cur_obj.get('Y')):
                    x, y = self.wcs.all_world2pix(ra=cur_obj.get('RA'), dec=cur_obj.get('DEC'))
                else:
                    x, y = [cur_obj.get('X'), cur_obj.get('Y')]

                if cur_obj['type'] == "Bubble" and cur_obj.get('expansion_velocity') <=0:
                    cur_obj['type'] = "Cloud"

                if not cur_obj.get('thickness'):
                    cur_obj['thickness'] = 1.

                if cur_obj['type'] == "Bubble":
                    generated_object = Bubble(xc=x, yc=y,
                                              max_brightness=cur_obj.get('max_brightness'),
                                              max_extinction=cur_obj.get('max_extinction'),
                                              spectral_axis=self.vel_grid,
                                              thickness=cur_obj['thickness'])
                elif cur_obj['type'] == "Cloud":
                    if not cur_obj.get('thickness'):
                        cur_obj['width'] = 0.1 * u.pc
                    generated_object = Cloud(xc=x, yc=y,
                                              max_brightness=cur_obj.get('max_brightness'),
                                              max_extinction=cur_obj.get('max_extinction'),
                                              thickness=cur_obj['thickness'])
                elif cur_obj['type'] == "Filament":
                    generated_object = Filament(xc=x, yc=y,
                                              max_brightness=cur_obj.get('max_brightness'),
                                              max_extinction=cur_obj.get('max_extinction'),
                                              width=cur_obj.get('width'))
            self.add_nebula(generated_object, cur_obj.get['spectrum_id'], zorder=cur_obj.get['zorder'])


#
#
#
# if __name__ == '__main__':
#     velocities = np.linspace(-150, 150, 10) << (u.km / u.s)
#     wcs = WCS()
#     ism = ISM(wcs)# nebulae=[Filament(), DIG()])
#     # print(ism.content['Brightness'])
#     # n = Bubble()
#     # print(ism.add_nebula(n))
#     bbl = Bubble(max_brightness=1e-15 * fluxunit, thickness=0.15, spectral_axis=velocities)
#     brt_2d = bbl.brightness_skyplane
#     # print(np.max(bbl.brightness_skyplane))
#     fig, ax = plt.subplots()
#     plt.imshow(brt_2d, origin='lower')
#     plt.colorbar()
#
#     fig, ax = plt.subplots()
#     plt.plot(velocities, bbl.spectrum())
#
#     plt.show()
