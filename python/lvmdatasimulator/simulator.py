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

from collections import OrderedDict
from scipy import special
from spectres import spectres
from astropy.io import ascii, fits
from dataclasses import dataclass
from astropy.convolution import Gaussian1DKernel, convolve

import lvmdatasimulator
from lvmdatasimulator.instrument import Spectrograph
from lvmdatasimulator.field import LVMField
from lvmdatasimulator.fibers import FiberBundle
from lvmdatasimulator.observation import Observation
from lvmdatasimulator.telescope import Telescope
from lvmdatasimulator import log, ROOT_DIR
from lvmdatasimulator.utils import round_up_to_odd
from joblib import Parallel, delayed
import os
import sys


@dataclass(frozen=True)
class Constants:
    h: u.erg * u.s = 6.6260755e-27 * u.erg * u.s  # Planck's constant in [erg*s]
    c: u.AA * u.s = 2.99792458e18 * u.AA / u.s  # Speed of light in [A/s]


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


def epp2flam(lam, fe, ddisp):
    """
    Convert photons per pixel [photons/s/cm2/pixel] to flux density [erg/s/cm2/A]

    Args:
        lam (array):
            wavelenght axis
        fe (array):
            spectrum in photons/s/cm2/pixel
        ddisp (float):
            dispersion in A/pix


    Returns:
        array:
            spectrum in erg/s/cm2/A
    """

    return fe * Constants.h * Constants.c / (lam * ddisp)


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

    return constant * 10 ** (exponent)


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

    kernel = Gaussian1DKernel(stddev=stddev.value, x_size=size.value)  # gaussian kernel
    return convolve(spectrum, kernel, boundary=boundary)


###################################################################################################


class Simulator:
    def __init__(
        self,
        source: LVMField,
        observation: Observation,
        spectrograph: Spectrograph,
        bundle: FiberBundle,
        telescope: Telescope,
        aperture: u.pix = 4 * u.pix,
        root: str = "./",
        overwrite: bool = True,
    ):

        self.source = source
        self.observation = observation
        self.spectrograph = spectrograph
        self.bundle = bundle
        self.telescope = telescope
        self.aperture = aperture
        self.root = root
        self.overwrite = overwrite

        # creating empty storage
        self.output_exposure = OrderedDict()
        self.output_exposure_noise = OrderedDict()
        # self.output_coadd = OrderedDict()

        self.outdir = os.path.join(self.root, "outputs")
        if os.path.isdir(self.outdir) and not self.overwrite:
            log.warning(f"{self.outdir} already exist. Terminating the simulation.")
            return
        if not os.path.isdir(self.outdir):
            os.mkdir(self.outdir)

    @functools.cached_property
    def extinction(self, extinction_file=f"{ROOT_DIR}/data/sky/LVM_LVM160_KLAM.dat"):
        """
        Returns atmospheric extinction coefficient sampled at instrumental wavelengths

        Args:
            extinction_file (str, optional):
                File containing the athmospheric extinction curve.
                Defaults to f'{ROOT_DIR}data/sky/LVM_LVM160_KLAM.dat'.
        """
        # log.info('Reading the atmospheric extinction from file.')
        self.extinction_file = extinction_file
        data = ascii.read(self.extinction_file)

        return self._resample_and_convolve(data["col1"], data["col2"])

    @functools.cached_property
    def sky(self):

        days_moon = self.observation.days_from_new_moon
        sky_file = f"{ROOT_DIR}/data/sky/LVM_{self.telescope.name}_SKY_{days_moon}.dat"
        area_fiber = np.pi * (self.bundle.fibers[0].diameter / 2)**2  # all fibers have same diam.
        data = ascii.read(sky_file)
        wave = data["col1"]
        brightness = data["col2"] * area_fiber.value  # converting to Fluxes from SBrightness

        return self._resample_and_convolve(wave, brightness,
                                           u.erg / (u.cm ** 2 * u.s * u.AA))

    def _resample_and_convolve(self, old_wave, old_flux, unit=None):
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
        disp0 = np.median(old_wave[1:-1] - old_wave[0:-2]) * u.AA / u.pix
        tmp_lam = np.arange(np.amin(old_wave), np.amax(old_wave), disp0.value)

        resampled_v0 = resample_spectrum(tmp_lam, old_wave, old_flux)
        out_spec = OrderedDict()
        for fiber in self.bundle.fibers:
            fiber_spec = OrderedDict()
            for branch in self.spectrograph.branches:
                # computing the FWHM of the final kernel (LSF (A) + fiber dispersion (pix))
                lsf_fwhm = branch.lsf_fwhm / disp0  # from A to pix

                # fiber dispersion must be converted to match the pixel scale of the tmp axis
                dfib_lam = fiber.dispersion * branch.wavecoord.step / disp0

                fwhm = np.sqrt((lsf_fwhm) ** 2 + (dfib_lam) ** 2)

                # do both convolutions at the same time
                convolved = convolve_for_gaussian(resampled_v0, fwhm, boundary="extend")
                resampled_v1 = resample_spectrum(branch.wavecoord.wave.value, tmp_lam, convolved)

                if unit:
                    fiber_spec[branch.name] = resampled_v1 * unit
                else:
                    fiber_spec[branch.name] = resampled_v1

            out_spec[fiber.id] = fiber_spec

        return out_spec

    @functools.cached_property
    def target_spectra(self):
        """Extract spectra of the terget from the field object"""

        obj_spec = OrderedDict()
        wl_grid = np.arange(3650, 9900.01, 0.06) * u.AA
        index, spectra = self.source.extract_spectra(self.bundle.fibers, wl_grid)
        log.info('Recovering target spectra.')
        for fiber in self.bundle.fibers:

            original = spectra[index == fiber.id, :][0]
            # from here, this is a replica of _resample_and_convolve()
            # I cannot use the method directly because I cannot use the same spectra for all fibers
            disp0 = np.median(wl_grid[1:-1] - wl_grid[0:-2])

            fiber_spec = OrderedDict()

            for branch in self.spectrograph.branches:

                lsf_fwhm = branch.lsf_fwhm / disp0  # from A to pix
                dfib_lam = fiber.dispersion * branch.wavecoord.step / disp0
                fwhm = np.sqrt((lsf_fwhm) ** 2 + (dfib_lam) ** 2)

                convolved = convolve_for_gaussian(original, fwhm, boundary="extend")
                resampled_v1 = resample_spectrum(branch.wavecoord.wave.value,
                                                 wl_grid.value, convolved)

                fiber_spec[branch.name] = resampled_v1 * (u.erg / (u.cm ** 2 * u.s * u.AA))
            obj_spec[fiber.id] = fiber_spec
        return obj_spec

    def _simulate_observations_single_fiber(self, fiber):
        spectrum = self.target_spectra[fiber.id]

        # convert spectra to electrons
        tmp_target = self._obj_to_electrons(spectrum, fiber.id)  # from units to electrons
        tmp_sky = self._sky_to_electrons(self.sky[fiber.id])

        # create 2D spectra
        tmp_target_2d = self._to_2d(fiber, tmp_target)
        tmp_sky_2d = self._to_2d(fiber, tmp_sky)
        tmp_noise_2d = self._make_noise(tmp_target_2d, tmp_sky_2d)

        # extract apertures
        single_exposure = self._extract_aperture(tmp_target_2d, tmp_noise_2d, tmp_sky_2d)

        # flux calibrate 1D spectra
        calibrated = self._flux_calibration(fiber.id, single_exposure)
        self.output_exposure[fiber.id] = calibrated

        # create a realistic spectrum with noise
        self.output_exposure_noise[fiber.id] = self._add_noise(single_exposure)
        return

    def simulate_observations(self):
        """
        Main function of the simulators. It takes everything we have done before, and simulate
        the data
        """
        log.info('Simulating observations.')

        if len(self.bundle.fibers) < 500:
            _ = [self._simulate_observations_single_fiber(fiber) for fiber in self.bundle.fibers]
        else:
            _ = Parallel(n_jobs=lvmdatasimulator.n_process)(
                delayed(self._simulate_observations_single_fiber)(fiber)
                for fiber in self.bundle.fibers
            )
        return

    def save_outputs(self):

        log.info('Saving the outputs:')
        for branch in self.spectrograph.branches:
            self._save_outputs_drp(branch)
            self._save_outputs_with_noise(branch)

        pass

    def _save_outputs_drp(self, branch):

        ids, signal, noise, sky = self._reorganize_to_rss_drp(branch)

        primary = self._create_primary_hdu(branch)

        signal_hdu = fits.ImageHDU(data=signal, name="FLUX")
        signal_hdu.header["BUNIT"] = "erg/(cm2 s A)"
        primary.header["EXT1"] = "FLUX"

        noise_hdu = fits.ImageHDU(data=noise, name="ERR")
        noise_hdu.header["BUNIT"] = "erg/(cm2 s A)"
        primary.header["EXT2"] = "ERR"

        sky_hdu = fits.ImageHDU(data=sky, name="SKY")
        sky_hdu.header["BUNIT"] = "erg/(cm2 s A)"
        primary.header["EXT3"] = "SKY"

        wave_hdu = fits.ImageHDU(data=branch.wavecoord.wave.value, name="WAVE")
        wave_hdu.header["BUNIT"] = "Angstrom"
        primary.header["EXT4"] = "WAVE"

        col1 = fits.Column(name="ID", format="10A", array=ids)
        ids_hdu = fits.BinTableHDU.from_columns([col1], name="FIBERID")
        primary.header["EXT5"] = "FIBERID"

        hdul = fits.HDUList([primary, signal_hdu, noise_hdu, sky_hdu, wave_hdu, ids_hdu])

        filename = f"{self.outdir}/{self.source.name}_{branch.name}_exposure.fits"
        hdul.writeto(filename, overwrite=True)
        log.info(f'{filename} saved.')

    def _save_outputs_with_noise(self, branch):

        ids, target, total, noise, sky, snr = self._reorganize_to_rss_el(branch)
        primary = self._create_primary_hdu(branch)

        target_hdu = fits.ImageHDU(data=target, name="TARGET")
        target_hdu.header["BUNIT"] = "e/pix"
        primary.header["EXT1"] = "TARGET"

        total_hdu = fits.ImageHDU(data=total, name="TOTAL")
        total_hdu.header["BUNIT"] = "e/pix"
        primary.header["EXT2"] = "TOTAL"

        noise_hdu = fits.ImageHDU(data=noise, name="ERR")
        noise_hdu.header["BUNIT"] = "e/pix"
        primary.header["EXT3"] = "ERR"

        stn_hdu = fits.ImageHDU(data=snr, name="SNR")
        stn_hdu.header["BUNIT"] = ""
        primary.header["EXT4"] = "SNR"

        sky_hdu = fits.ImageHDU(data=sky, name="SKY")
        sky_hdu.header["BUNIT"] = "e/pix"
        primary.header["EXT5"] = "SKY"

        wave_hdu = fits.ImageHDU(data=branch.wavecoord.wave.value, name="WAVE")
        wave_hdu.header["BUNIT"] = "Angstrom"
        primary.header["EXT6"] = "WAVE"

        col1 = fits.Column(name="ID", format="10A", array=ids)
        ids_hdu = fits.BinTableHDU.from_columns([col1], name="FIBERID")
        primary.header["EXT7"] = "FIBERID"

        hdul = fits.HDUList([primary, target_hdu, total_hdu, noise_hdu, stn_hdu, sky_hdu,
                             wave_hdu, ids_hdu])

        filename = f"{self.outdir}/{self.source.name}_{branch.name}_exposure_electron.fits"
        hdul.writeto(filename, overwrite=True)
        log.info(f'{filename} saved.')

    def _obj_to_electrons(self, spectrum, fiber_id):

        # Number of object electrons per spectral pixel
        # Returns the total number of electrons in a spectral pixel of width ddisp.
        # Assumes all electrons fall in one spatial pixel so the output must then
        # be redistribute across the spatial PSF.

        out = OrderedDict()
        for branch in self.spectrograph.branches:

            # convert spectrum to electrons
            spectrum_e = flam2epp(branch.wavecoord.wave, spectrum[branch.name],
                                  branch.wavecoord.step)

            # compute constant
            constant = self.observation.exptime * branch.efficiency * self.telescope.aperture_area

            # atmospheric extinction
            atmosphere = self.extinction[fiber_id][branch.name] * self.observation.airmass

            # put everything together
            out[branch.name] = spectrum_e * constant * 10 ** (-0.4 * (atmosphere)) * u.electron

        return out

    def _to_2d(self, fiber, spectrum):
        """ Transforming 1D to 2D. Not sure what is happening"""

        nypix = int(round_up_to_odd(fiber.nypix.value))

        out = OrderedDict()
        for branch in self.spectrograph.branches:
            new_2d_array = np.zeros((len(spectrum[branch.name]), nypix))
            ne_frac = np.zeros(nypix)
            # not sure what is happening here
            for i in range(nypix):
                j = i - int(np.floor(nypix / 2))
                ymin = (j - 0.5) / np.sqrt(2) / (fiber.dispersion / 2.355)
                ymax = (j + 0.5) / np.sqrt(2) / (fiber.dispersion / 2.355)
                # integral of a normalize gaussian between ymin and ymax
                ne_frac[i] = (special.erf(ymax.value) - special.erf(ymin.value)) / 2

            for i, factor in enumerate(ne_frac):
                new_2d_array[:, i] = factor * spectrum[branch.name]
            out[branch.name] = new_2d_array * spectrum[branch.name].unit  # fixing the unit

        return out

    def _sky_to_electrons(self, sky_spectrum):

        out = OrderedDict()
        for branch in self.spectrograph.branches:
            # convert spectrum to electrons
            sky_e = flam2epp(branch.wavecoord.wave, sky_spectrum[branch.name],
                             branch.wavecoord.step)

            # compute constant
            constant = self.observation.exptime * branch.efficiency * self.telescope.aperture_area
            out[branch.name] = sky_e * constant * u.electron

        return out

    def _make_noise(self, spectrum, sky):
        """ compute the noise. Can work with any array size"""

        out = OrderedDict()

        # this is to create more realistic noise
        # ron = np.random.randn(spectrum[branch.name].shape) * branch.ron

        for branch in self.spectrograph.branches:
            dark = branch.dark * self.observation.exptime
            out[branch.name] = np.sqrt(spectrum[branch.name].value + sky[branch.name].value +
                                       branch.ron.value ** 2 + dark.value) * \
                spectrum[branch.name].unit
        return out

    def _extract_aperture(self, spec2d, noise2d, sky2d):

        stnout = OrderedDict()
        sout = OrderedDict()
        nout = OrderedDict()
        skyout = OrderedDict()

        for branch in self.spectrograph.branches:
            # they all have the same size
            npix = spec2d[branch.name].shape[0]

            # finding the offset from the central pixel
            dy = np.arange(spec2d[branch.name].shape[1]) - \
                np.floor(spec2d[branch.name].shape[1] / 2.0)

            # selecting pixels < of half the size of the aperture
            sely = np.abs(dy) <= self.aperture.value / 2.0
            sely = np.repeat(sely[np.newaxis, :], npix, axis=0)  # extending to full spectrum

            # extracting the apertures
            # creating helper objects to keep working with arrays
            faux = np.zeros(sely.shape)
            eaux = np.zeros(sely.shape)
            saux = np.zeros(sely.shape)
            faux[sely] = spec2d[branch.name][sely]  # spectrum
            eaux[sely] = noise2d[branch.name][sely]  # noise
            saux[sely] = sky2d[branch.name][sely]  # sky
            signal = faux.sum(axis=1)
            noise = np.sqrt(np.sum(eaux ** 2, axis=1))
            sky = saux.sum(axis=1)
            stnout[branch.name] = signal / noise
            sout[branch.name] = signal
            nout[branch.name] = noise
            skyout[branch.name] = sky

        return {"spectrum": sout, "noise": nout, "snr": stnout, "sky": skyout}

    def _add_noise(self, exposure):

        signal_noise = OrderedDict()
        obj_noise = OrderedDict()

        for branch in self.spectrograph.branches:

            noise_to_add = np.random.standard_normal(exposure["noise"][branch.name].shape) *\
                exposure["noise"][branch.name]

            signal = exposure["spectrum"][branch.name] + exposure["sky"][branch.name]
            signal_with_noise = signal + noise_to_add
            target = signal_with_noise - exposure["sky"][branch.name]
            signal_noise[branch.name] = signal_with_noise
            obj_noise[branch.name] = target

        return {"signal": signal_noise, "target": obj_noise, "noise": exposure["noise"],
                "sky": exposure["sky"], "snr": exposure["snr"]}

    def _coadd(self, spectrum, sky, noise):

        spec_coadd = OrderedDict()
        sky_coadd = OrderedDict()
        noise_coadd = OrderedDict()

        for branch in self.spectrograph.branches:

            spec_coadd[branch.name] = spectrum[branch.name] * self.observation.nexp
            sky_coadd[branch.name] = sky[branch.name] * self.observation.nexp
            noise_coadd[branch.name] = noise[branch.name] * np.sqrt(self.observation.nexp)

        return spec_coadd, sky_coadd, noise_coadd

    def _reorganize_to_rss_drp(self, branch):

        nfibers = self.bundle.nfibers
        fib_id = []

        signal = np.zeros((nfibers, branch.wavecoord.npix))
        noise = np.zeros((nfibers, branch.wavecoord.npix))
        sky = np.zeros((nfibers, branch.wavecoord.npix))

        for i, (fiber_id, spectra) in enumerate(self.output_exposure.items()):

            fib_id.append(fiber_id)
            signal[i, :] = spectra["spectrum"][branch.name]
            noise[i, :] = spectra["noise"][branch.name]
            sky[i, :] = spectra["sky"][branch.name]

        return np.array(fib_id), signal, noise, sky

    def _reorganize_to_rss_el(self, branch):

        nfibers = self.bundle.nfibers
        fib_id = []

        # outputs with noise
        target_w_noise = np.zeros((nfibers, branch.wavecoord.npix))
        total_w_noise = np.zeros((nfibers, branch.wavecoord.npix))
        sky_el = np.zeros((nfibers, branch.wavecoord.npix))
        noise_el = np.zeros((nfibers, branch.wavecoord.npix))
        snr = np.zeros((nfibers, branch.wavecoord.npix))

        for i, (fiber_id, spectra) in enumerate(self.output_exposure_noise.items()):
            fib_id.append(fiber_id)
            target_w_noise[i, :] = spectra["target"][branch.name]
            total_w_noise[i, :] = spectra["signal"][branch.name]
            noise_el[i, :] = spectra["noise"][branch.name]
            sky_el[i, :] = spectra["sky"][branch.name]
            snr[i, :] = spectra["snr"][branch.name]

        return np.array(fib_id), target_w_noise, total_w_noise, noise_el, sky_el, snr

    def _create_primary_hdu(self, branch):

        primary = fits.PrimaryHDU()
        primary.header["TARGET"] = self.source.name
        primary.header["RA"] = self.source.ra.value
        primary.header["DEC"] = self.source.dec.value
        primary.header["AZ"] = (self.observation.target_coords_altaz.az.value,
                                "Azimuth of the target")
        primary.header["ALT"] = (self.observation.target_coords_altaz.alt.value,
                                 "Altitude of the target")
        primary.header["AIRMASS"] = self.observation.airmass
        primary.header["MJD"] = (self.observation.mjd, "MJD at start")
        primary.header["EXPTIME"] = self.observation.exptime.value
        primary.header["BRANCH"] = branch.name
        primary.header["MOON"] = (self.observation.moon_distance.value,
                                  "Fractional moon illumination")
        primary.header["DAY-MOON"] = (self.observation.days_from_new_moon,
                                      "Days from new moon")

        return primary

    def _flux_calibration(self, fiber_id, exposure):

        stnout = OrderedDict()
        sout = OrderedDict()
        nout = OrderedDict()
        skyout = OrderedDict()
        for branch in self.spectrograph.branches:
            # this remove the signature of the instruments and goes back to the real spectrum
            sout[branch.name] = self.target_spectra[fiber_id][branch.name]
            skyout[branch.name] = self.sky[fiber_id][branch.name]

            # this remove the signature of the instruments and goes back to the real spectrum
            constant = self.observation.exptime * branch.efficiency * self.telescope.aperture_area
            # constant2 = self.extinction[fiber_id][branch.name] * self.observation.airmass
            # applying telluric correction to noise
            tmp_noise = exposure['noise'][branch.name] / (constant)  # * 10 ** (-0.4 * constant2))

            nout[branch.name] = epp2flam(branch.wavecoord.wave, tmp_noise, branch.wavecoord.step)
            stnout[branch.name] = exposure['snr'][branch.name]

        return {"spectrum": sout, "noise": nout, "snr": stnout, "sky": skyout}
