import scipy.optimize
from astropy import units as u
import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import nquad
from tqdm import tqdm
from scipy.special import sph_harm
from dataclasses import dataclass
import functools
from scipy.ndimage.interpolation import map_coordinates
from scipy.interpolate import interp1d

@dataclass
class Filament:
    """
    Class of an isotropic cylindrical shape filament.
    Defined by its position, lenght, PA, radius, maximal optical depth
    if it is emission-type filament, then maximal brightness
    """
    pass

@dataclass
class DIG:
    """
    Class defining the DIG component. For now it is defined just by its brightness (constant)
    """
    pass

@dataclass
class Cloud:
    """Class of an isotropic spherical gas cloud without any ionization source.
    Defined by its position, radius, density, maximal optical depth"""
    radius: u.pc = 1.0 * u.pc
    pxscale: u.pc = 0.01 * u.pc
    phi_bins: int = 180
    theta_bins: int = 180
    rad_bins: int = 100
    total_brightness = 1.8e21 *u.cm**-2
    thickness = 1.0
    @functools.cached_property
    def theta_grid(self):
        return np.linspace(0, np.pi, self.theta_bins)

    @functools.cached_property
    def phi_grid(self):
        return np.linspace(0, 2 * np.pi, self.phi_bins)

    @functools.cached_property
    def rad_grid(self):
        return np.linspace(0, self.radius, self.rad_bins)

    @functools.cached_property
    def cartesian_1d_grid(self):
        npix = np.ceil(1.05*self.radius/self.pxscale).astype(int)
        return np.linspace(-npix, npix, 2*npix+1)*self.pxscale

    @functools.cached_property
    def brightness_3d_spherical(self):
        """
        Method to calculate brightness (or opacity) of the cloud at given theta, phi and radii

        theta: float -- polar angle [0, np.pi]
        phi: float -- azimuthal angle [0, 2 * np.pi]
        rad: float -- radius [0, self.radius]
        Returns:
            3D cube of brightness in theta-phi-rad grid, in astropy.Quantity
        """
        theta, phi, rho = np.meshgrid(self.theta_grid, self.phi_grid, self.rad_grid)
        brt=np.ones_like(theta)
        brt[rho<(self.radius*(1-self.thickness))]=0
        brt[rho > self.radius]=0
        brt=brt/np.sum(brt)*self.total_brightness
        return brt

    def brightness_3d_cartesian(self):
        X, Y, Z = np.meshgrid(self.cartesian_1d_grid,
                              self.cartesian_1d_grid,
                              self.cartesian_1d_grid)
        Pc = (np.arctan2(Y, X))
        Rc = (np.sqrt(X ** 2 + Y ** 2 + Z ** 2))
        Rc[Rc==0*u.pc] = 1e-3*self.pxscale
        Tc = (np.arccos(Z/Rc))

        Pc[Pc < (0*u.radian)] = 2 * np.pi*u.radian + Pc[Pc < (0*u.radian)]
        Pc[Pc > (2 * np.pi*u.radian)] = -2 * np.pi*u.radian + Pc[Pc > (2 * np.pi*u.radian)]
        Tc[Tc < (0*u.radian)] = np.pi*u.radian + Tc[Tc < (0*u.radian)]
        Tc[Tc > (np.pi*u.radian)] = -np.pi*u.radian + Tc[Tc > (np.pi*u.radian)]

        ir = interp1d(self.rad_grid, np.arange(self.rad_bins), bounds_error=False)
        ith = interp1d(self.theta_grid, np.arange(self.theta_bins))
        iphi = interp1d(self.phi_grid, np.arange(self.phi_bins))
        new_ir = ir(Rc.ravel())
        new_ith = ith(Tc.ravel())
        new_iphi = iphi(Pc.ravel())
        cart_data = map_coordinates(self.brightness_3d_spherical,
                                    np.vstack([new_ith, new_iphi, new_ir]),
                                    order=1, mode='constant', cval=0)
        cart_data=cart_data/np.sum(cart_data)*np.sum(self.brightness_3d_spherical)
        # The data is reshaped and returned
        return cart_data.reshape([len(self.cartesian_1d_grid)]*3)

    def brightness_skyplane(self, ):
        return np.nansum(self.brightness_3d_cartesian(),0)


@dataclass
class Bubble(Cloud):
    """Class of an isotropic thin expanding bubble."""
    spectral_axis: u.km / u.s = None
    expansion_velocity: u.km / u.s = 20 * u.km / u.s
    turbulent_sigma: u.km / u.s = 30 * u.km / u.s
    center_velocity: u.km / u.s = 0 * u.km / u.s
    total_brightness = 1e-13 * u.erg/u.cm**2 / u.s ** 2
    thickness: float = 0.2
    quad_epsrel = 1e-2

    # def velocity(self, theta: float, phi: float) -> u.km / u.s:
    #     """
    #     Method to calculate expansion velocity at given theta, phi
    #
    #     Args:
    #         theta: float -- polar angle [0, np.pi]
    #         phi: float -- azimuthal angle [0, 2 * np.pi]
    #
    #     Returns:
    #         Expansion velocity, in astropy.Quantity
    #     """
    #     return self.expansion_velocity * np.cos(theta)
    #
    #
    # def d_area(self, theta: float, phi: float, rad:u.pc):
    #     """Area differential dS/dφdθ, in unitless pc**2"""
    #     return rad.to_value(u.pc) ** 2 * np.sin(theta)
    #
    #
    # def area(self):
    #     return nquad(self.d_area, [[0, np.pi], [0, 2 * np.pi]])
    #
    # def turbulent_lsf(self, theta, phi, velocity):
    #     """Line spread function as a function of coorinates, including the velocity center shift"""
    #     mu = self.velocity(theta, phi)
    #     sig = self.turbulent_sigma
    #     return 1. / (np.sqrt(2. * np.pi) * sig) * np.exp(-np.power((velocity - mu) / sig, 2.) / 2)
    #
    # def d_spectrum(self, theta: float, phi: float, rad:u.pc, velocity: u.km / u.s):
    #     """Returns local spectrum, per pc**2 of area"""
    #     return (
    #             self.brightness_3d_spherical[theta, phi, rad]
    #             * self.d_area(theta, phi, rad) * u.pc ** 2
    #             * self.turbulent_lsf(theta, phi, velocity)
    #     ).to(u.W / (u.km / u.s))
    #
    # def _d_spectrum_dimensionless(self, *args):
    #     return self.d_spectrum(*args).to_value(u.W / (u.km / u.s))
    #
    # def spectrum(self) -> u.Jy:
    #     velocities = self.spectral_axis.to(u.km / u.s, equivalencies=u.spectral())
    #     theta, phi, rad, velocities = np.meshgrid(self.theta_grid, self.phi_grid,
    #                                               self.rad_grid,velocities)
    #     spectrum = (
    #             np.sum(self.d_spectrum(theta, phi, rad, velocities), axis=(0, 1))
    #             * (self.theta_grid[1] - self.theta_grid[0])
    #             * (self.phi_grid[1] - self.phi_grid[0])
    #     )
    #     e_spectrum = spectrum * 0
    #     return spectrum, e_spectrum


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


if __name__ == '__main__':
    velocities = np.linspace(-120, 120, 240) << (u.km / u.s)
    bbl = Bubble(thickness=0.1)
    brt_2d = bbl.brightness_skyplane()

    fig, ax = plt.subplots()
    plt.imshow(brt_2d,origin='lower')
    plt.colorbar()
    plt.show()

