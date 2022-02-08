# encoding: utf-8
#
# @Author: Oleg Egorov, Enrico Congiu
# @Date: Nov 12, 2021
# @Filename: simulator.py
# @License: BSD 3-Clause
# @Copyright: Oleg Egorov, Enrico Congiu

# Adapted from Local Volume Mapper (LVM) Exposure Time Calculator
# adapted from the Las Campanas Observatory ETC
# original files: https://github.com/gblancm/lcoetc
#
# 2017-08-08: This is a BETA Version written by Guillermo A. Blanc. It has not been validated.
# Use at your own risk.
# 2020-09-15: Ported from a cgi web interface to a stand alone python script by Kathryn Kreckel.
# 2020-09-28: New zeropoints and bug fixes ported from Guillermo's code
# 2021-04-08: added 'run_lvmetc' function to facilitate importing this as a package

import numpy as np
import astropy.units as u
import functools
import scipy

from spectres import spectres
from astropy.io import ascii
from dataclasses import dataclass
from astropy.convolution import Gaussian1DKernel, convolve

from lvmdatasimulator.instrument import Spectrograph
from lvmdatasimulator.field import LVMField
from lvmdatasimulator.observation import Observation
from lvmdatasimulator.telescope import Telescope
from lvmdatasimulator import log, ROOT_DIR
from lvmdatasimulator.utils import round_up_to_odd

import os


@dataclass(frozen=True)
class Constants:
    h: u.erg * u.s = 6.6260755e-27 * u.erg * u.s    # Planck's constant in [erg*s]
    c: u.A * u.s = 2.99792458e18 * u.A * u.s   # Speed of light in [A/s]


def flam2epp(lam, flam, ddisp):
    """
    Convert flux density [erg/s/cm2/A] to photons per pixel [photons/s/cm2/pixel]

    Args:
        lam (array-like):
            wavelength array associated to the spectrum
        flam (array-like):
            spectrum in units of erg/s/cm2/A
        ddisp (float):
            the pixel scale in A/pixel

    Returns:
        array-like:
            spectrum converted to photons/s/cm2/pixel
    """
    return flam * lam * ddisp / (Constants.h * Constants.c)


def zp2elam(zp, lam, neobj, ddisp, texp, atel, klam, airmass):
    """
    Claculates the system efficiency "elam" from a Zero Point "zp" (AB mag) corresponding to
    "neobj" electrons per pixel of width "ddisp" collected in "texp" at airmass "amass"
    The ZP is typically given for 1 e/s/pixel at amass=1 and sometimes 1 ADU/s/pixel at amass=1

    Args:
        zp (float ?):
            zeropoint of the instrument.
        lam (array-like):
            ?
        neobj ([type]):
            Not sure what this is. Number of electron produced by the object?
        ddisp (float):
            dispersion of the instrumment
        texp (float):
            Exposure time
        atel ([type]):
            collecting area of the telescope
        klam ([type]):
            extinction as a function of lambda ?
        airmass (float):
            airmass

    Returns:
        ?:
            efficiency of the telescope
    """

    exponent = 0.4 * (zp + 48.6 + klam * airmass)
    constant = Constants.h * lam * neobj / (ddisp * texp * atel)

    return constant * 10**(exponent)


def elam2zp(elam, lam, neobj, ddisp, texp, atel, klam, airmass):
    """
    Zero Point from System Efficiency. It's the inverse function of zp2elam.

    Args:
        elam ([type]):
            system efficiency
        lam ([type]):
            wavelengths
        neobj ([type]):
            [description]
        ddisp ([type]):
            [description]
        texp ([type]):
            [description]
        atel ([type]):
            [description]
        klam ([type]):
            [description]
        amass ([type]):
            [description]

    Returns:
        [type]: [description]
    """
    const1 = elam * ddisp * texp * atel
    const2 = Constants.h * lam * neobj

    return 2.5 * np.log10(const1 / const2) - 48.6 - klam * airmass


def resample_spectrum(new_wave, old_wave, flux):
    """
    Resample spectrum to a new wavelength grid using the spectres package.

    Args:
        new_wave (array-like):
            new wavelength axis.
        old_wave (array-like):
            original wavelength axis
        flux (array-like):
            original spectrum

    Returns:
        array-like:
            spectrum resampled onto the new_wave axis
    """

    resampled = spectres(new_wave, old_wave, flux)

    return resampled


def convolve_for_gaussian(spectrum, fwhm, boundary):
    """
    Convolve a spectrum for a Gaussian kernel.

    Args:
        spectrum (array):
            spectrum to be convolved.
        fwhm (float):
            FWHM of the gaussian kernel.
        boundary (str):
            flag indicating how to handle boundaries.

    Returns:
        array:
            convolved spectrum
    """

    stddev = fwhm / 2.355  # from fwhm to sigma
    size = round_up_to_odd(stddev)  # size of the kernel

    kernel = Gaussian1DKernel(stddev=stddev, x_size=size)  # gaussian kernel for convolution
    return convolve(spectrum, kernel, boundary=boundary)


###################################################################################################


class Simulator:

    def __init__(self, source: LVMField,
                 observation: Observation,
                 spectrograph: Spectrograph,
                 telescope: Telescope,
                 aperture: u.pix = 4 * u.pix,
                 root: str = './',
                 overwrite: bool = True):

        self.source = source
        self.observation = observation
        self.spectrograph = spectrograph
        self.telescope = telescope
        self.aperture = aperture
        self.root = root
        self.overwrite = overwrite

        # creating empty storage
        self.target = {}
        self.target_2d = {}
        self.sky_1d = {}
        self.sky_2d = {}
        self.noise_2d = {}
        self.output_exposure = {}
        self.target_2d_coadd = {}
        self.sky_2d_coadd = {}
        self.noise_2d_coadd = {}
        self.output_coadd = {}

        self.outdir = os.path.join(self.root, 'outputs')
        if os.path.isdir(self.outdir) and not self.overwrite:
            log.warning(f'{self.outdir} already exist. Terminating the simulation.')
            return

    @functools.cached_property
    def extinction(self, extinction_file=f'{ROOT_DIR}data/sky/LVM_LVM160_KLAM.dat'):
        """
        Returns atmospheric extinction coefficient sampled at instrumental wavelengths

        Args:
            extinction_file (str, optional):
                File containing the athmospheric extinction curve.
                Defaults to f'{ROOT_DIR}data/sky/LVM_LVM160_KLAM.dat'.
        """
        self.extinction_file = extinction_file
        data = ascii.read(self.extinction_file)

        return self._resample_and_convolve(data['col1'], data['col2'])

    @functools.cached_property
    def sky(self):

        days_moon = self.observation.days_from_new_moon()
        sky_file = f'{ROOT_DIR}/LVM_{self.instrument.name}_SKY_{days_moon}.dat'
        data = ascii.read(sky_file)

        return self._resample_and_convolve(data['col1'], data['col2'])

    def _resample_and_convolve(self, old_wave, old_flux):
        """
        auxiliary function to resample a spectrum to the instrument wavelength array and
        convolve it for the LSF and for the fiber_profile

        Args:
            old_wave (array-like):
                original wavelength axis
            old_flux (array-like):
                original spectrum

        Returns:
            array-like:
                spectrum resampled to instrument wavelength axis and convolved for lsf and fiber
        """

        # creating an intermediate wavelength axis regularly sampled with the same
        # dispersion of the original one
        disp0 = np.median(old_wave[1: - 1] - old_wave[0: - 2])
        tmp_lam = np.arange(np.amin(old_wave), np.amax(old_wave), disp0)

        resampled_v0 = resample_spectrum(tmp_lam, old_wave, old_flux)
        out_spec = {}
        for fiber in self.spectrograph.bundle.fibers:
            fiber_spec = {}
            for branch in self.spectrograph.branches:
                # computing the FWHM of the final kernel (LSF (A) + fiber dispersion (pix))
                lsf_fwhm = branch.lsf_fwhm / disp0  # from A to pix
                # fiber dispersion must be converted to match the pixel scale of the tmp axis
                dfib_lam = fiber.dispersion * branch.wavecoord.step / disp0
                fwhm = np.sqrt((lsf_fwhm)**2 + (dfib_lam)**2)

                # do both convolutions at the same time
                convolved = convolve_for_gaussian(resampled_v0, fwhm, boundary='extend')
                resampled_v1 = resample_spectrum(branch.wavecoord.wave, tmp_lam, convolved)

                fiber_spec[branch.name] = resampled_v1

            out_spec[fiber.id] = fiber_spec

        return out_spec

    @functools.cached_property
    def target_spectra(self):
        """Extract spectra of the terget from the field object"""

        obj_spec = {}
        for fiber in self.spectrograph.bundle.fibers:

            # extract spectrum from the source. The method is there, but I do not know how
            # it work yet. this is therefore a placeholder, that should be fixed in the future
            # I'm assuming the output to be a dictionary with 'wave' and 'spectrum'
            tmp_spec = self.source.get_spectrum(fiber)

            # from here, this is a replica of _resample_and_convolve()
            # I cannot use the method directly because I cannot use the same spectra for all fibers
            disp0 = np.median(tmp_spec['wave'][1: -1] - tmp_spec['wave'][0: -2])
            tmp_lam = np.arange(np.amin(tmp_spec['wave']), np.amax(tmp_spec['wave']), disp0)

            resampled_v0 = resample_spectrum(tmp_lam, tmp_spec['wave'], tmp_spec['spectrum'])

            fiber_spec = {}

            for branch in self.spectrograph.branches:

                lsf_fwhm = branch.lsf_fwhm / disp0  # from A to pix
                dfib_lam = fiber.dispersion * branch.wavecoord.step / disp0
                fwhm = np.sqrt((lsf_fwhm)**2 + (dfib_lam)**2)

                convolved = convolve_for_gaussian(resampled_v0, fwhm, boundary='extend')
                resampled_v1 = resample_spectrum(branch.wavecoord.wave, tmp_lam, convolved)

                fiber_spec[branch.name] = resampled_v1
            obj_spec[fiber.id] = fiber_spec
        return obj_spec

    def simulate_observations(self):
        """
        Main function of the simulators. It takes everything we have done before, and simulate
        the data
        """

        for fiber in self.spectrograph.bundle.fibers:
            spectrum = self.target_spectra[fiber.id]

            # convert spectra to electrons
            tmp_target = self._obj_to_electrons(spectrum)  # from units to electrons
            self.target[fiber.id] = tmp_target  # store it

            tmp_sky = self._sky_to_electrons(self.sky[fiber.id])
            self.sky_1d[fiber.id] = tmp_sky

            # create 2D spectra
            tmp_target_2d = self._to_2d(fiber, spectrum)
            self.target_2d[fiber.id] = tmp_target_2d

            tmp_sky_2d = self._to_2d(fiber, tmp_sky)
            self.sky_2d[fiber.id] = tmp_sky_2d

            tmp_noise_2d = self._make_noise(tmp_target_2d, tmp_sky_2d)
            self.noise_2d[fiber.id] = tmp_noise_2d

            # extract apertures
            single_exposure = self._extract_aperture(tmp_target_2d,
                                                     tmp_noise_2d,
                                                     tmp_sky_2d)
            self.output_exposure[fiber.id] = single_exposure

            # coadding
            tmp_target_2d_coadd = tmp_target_2d * self.observation.nexp
            self.target_2d_coadd[fiber.id] = tmp_target_2d_coadd

            tmp_sky_2d_coadd = tmp_sky_2d * self.observation.nexp
            self.sky_2d_coadd[fiber.id] = tmp_sky_2d_coadd

            tmp_noise_2d_coadd = tmp_noise_2d * np.sqrt(self.observation.nexp)
            self.noise_2d_coadd[fiber.id] = tmp_noise_2d_coadd

            coadded = self._extract_aperture(tmp_target_2d_coadd,
                                             tmp_noise_2d_coadd,
                                             tmp_sky_2d_coadd)
            self.output_coadd[fiber.id] = coadded
        return

    def _obj_to_electrons(self, spectrum):

        # Number of object electrons per spectral pixel
        # Returns the total number of electrons in a spectral pixel of width ddisp.
        # Assumes all electrons fall in one spatial pixel so the output must then
        # be redistribute across the spatial PSF.

        out = {}
        for branch in self.spectrograph.branches:

            # convert spectrum to electrons
            spectrum_e = flam2epp(branch.wavecoord.wave,
                                  spectrum[branch.name],
                                  branch.wavecoord.step)

            # compute constant
            constant = self.observation.exptime * branch.efficiency * self.telescope.aperture_area

            # atmospheric extinction
            atmosphere = self.extinction * self.observation.airmass

            # put everything together
            out[branch.name] = spectrum_e * constant * 10**(-0.4 * (atmosphere))

        return out

    def _to_2d(self, fiber, spectrum):
        """ Transforming 1D to 2D. Not sure what is happening"""

        nypix = int(round_up_to_odd(fiber.nypix))

        out = {}
        for branch in self.spectrograph.branches:
            new_2d_array = np.zeros((len(spectrum[branch.name]), nypix))
            ne_frac = np.zeros(nypix)
            # not sure what is happening here
            for i in range(nypix):
                j = i - int(np.floor(nypix / 2))
                ymin = (j - 0.5) / np.sqrt(2) / (fiber.dispersion / 2.355)
                ymax = (j + 0.5) / np.sqrt(2) / (fiber.dispersion / 2.355)
                ne_frac[i] = (scipy.special.erf(ymax) - scipy.special.erf(ymin)) / 2

            for i, factor in enumerate(nypix):
                new_2d_array[:, i] = factor * spectrum[branch.name]
            out[branch.name] = new_2d_array

        return out

    def _sky_to_electrons(self, sky_spectrum):

        out = {}
        for branch in self.spectrograph.branches:
            # convert spectrum to electrons
            sky_e = flam2epp(branch.wavecoord.wave,
                             sky_spectrum[branch.name],
                             branch.wavecoord.step)

            # compute constant
            constant = self.observation.exptime * branch.efficiency * self.telescope.aperture_area
            out[branch.name] = sky_e * constant

        return out

    def _make_noise(self, spectrum, sky):
        """ compute the noise. Can work with any array size"""

        out = {}

        for branch in self.spectrograph.brances:
            out[branch.name] = np.sqrt(spectrum[branch.name] + sky[branch.name] + branch.ron**2 +
                                       branch.dark * self.observation.exptime)

        return out

    def _extract_aperture(self, spec2d, noise2d, sky2d):

        stnout = {}
        sout = {}
        nout = {}
        skyout = {}

        for branch in self.spectrograph.branches:
            # they all have the same size
            npix = spec2d[branch.name].shape[0]

            # finding the offset from the central pixel
            dy = np.arange(spec2d[branch.name].shape[1]) - \
                np.floor(spec2d[branch.name].shape[1] / 2.0)

            # selecting pixels < of half the size of the aperture
            sely = np.abs(dy) <= self.aperture / 2.0
            sely = np.repeat(sely[np.newaxis, :], npix, axis=0)  # extending to full spectrum

            # extracting the apertures
            faux = spec2d[branch.name][sely]  # spectrum
            eaux = noise2d[branch.name][sely]  # noise
            saux = sky2d[branch.name][sely]  # sky
            signal = faux.sum(axis=1)
            noise = np.sqrt(np.sum(eaux ** 2, axis=1))
            sky = saux.sum(axis=1)
            stnout[branch.name] = signal / noise
            sout[branch.name] = signal
            nout[branch.name] = noise
            skyout[branch.name] = sky

        return {'spectrum': sout, 'noise': nout, 'snr': stnout, 'sky': skyout}
