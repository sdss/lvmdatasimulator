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

from multiprocessing import Pool
from collections import OrderedDict
from scipy import special
from scipy.interpolate import interp1d
from spectres import spectres
from astropy.io import ascii, fits
from astropy.table import vstack, Table
from dataclasses import dataclass
from astropy.convolution import Gaussian1DKernel, convolve

import lvmdatasimulator
from lvmdatasimulator.instrument import Spectrograph
from lvmdatasimulator.field import LVMField
from lvmdatasimulator.fibers import FiberBundle
from lvmdatasimulator.observation import Observation
from lvmdatasimulator.telescope import Telescope
from lvmdatasimulator import log
from lvmdatasimulator.utils import round_up_to_odd, set_geocoronal_ha, open_sky_file
from joblib import Parallel, delayed
import os
# import sys


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


def resample_spectrum(new_wave, old_wave, flux, fast=True):
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
    if fast:
        f = interp1d(old_wave, flux, fill_value='extrapolate')
        resampled = f(new_wave)
    else:
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
        aperture: u.pix = 10 * u.pix,
        root: str = lvmdatasimulator.WORK_DIR,
        overwrite: bool = True,
        fast: bool = True
    ):

        self.source = source
        self.observation = observation
        self.spectrograph = spectrograph
        self.bundle = bundle
        self.telescope = telescope
        self.aperture = aperture
        if root == lvmdatasimulator.WORK_DIR:
            subdir = source.name
        else:
            subdir = ''
        self.root = os.path.join(root, subdir)
        self.overwrite = overwrite
        self.fast = fast

        # creating empty storage
        self.output_no_noise = OrderedDict()  # realization without noise
        self.output_noise = OrderedDict()  # realization with noise
        self.output_calib = OrderedDict()  # flux calibrated
        # self.output_coadd = OrderedDict()

        self.outdir = os.path.join(self.root, "outputs")
        if os.path.isdir(self.outdir) and not self.overwrite:
            log.warning(f"{self.outdir} already exist. Terminating the simulation.")
            return
        if not os.path.isdir(self.outdir):
            os.makedirs(self.outdir)

        self.extinction = None
        self.sky = None
        self.target_spectra = None

    def extract_extinction(self, extinction_file=os.path.join(lvmdatasimulator.DATA_DIR, 'sky',
                                                              'LVM_LVM160_KLAM.dat')):
        """
        Returns atmospheric extinction coefficient sampled at instrumental wavelengths

        Args:
            extinction_file (str, optional):
                File containing the athmospheric extinction curve.
                Defaults to f'{DATA_DIR}/sky/LVM_LVM160_KLAM.dat'.
        """

        log.info('Reading the atmospheric extinction from file.')
        self.extinction_file = extinction_file
        data = ascii.read(self.extinction_file)

        log.info('Resample extinction file to instrument wavelength solution.')
        return self._resample_and_convolve(data["col1"], data["col2"])

    def extract_sky(self):
        """
        Return sky emission spectrum sampled at instrumental wavelengths
        """
        wave, brightness = open_sky_file(self.observation.sky_template, self.observation.days_moon,
                                         self.telescope.name)

        if self.observation.geocoronal is not None:
            ha = self.observation.geocoronal
            brightness = set_geocoronal_ha(wave, brightness, ha)

        area_fiber = np.pi * (self.bundle.fibers_science[0].diameter / 2) ** 2  # all fibers same diam.
        flux = brightness * area_fiber.value  # converting to Fluxes from SBrightness

        log.info('Resample sky emission to instrument wavelength solution.')
        return self._resample_and_convolve(wave, flux, u.erg / (u.cm ** 2 * u.s * u.AA))

    def _resample_and_convolve(self, old_wave, old_flux, unit=None):
        """
        Auxiliary function to resample a spectrum to the instrument wavelength array and
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

        resampled_v0 = resample_spectrum(tmp_lam, old_wave, old_flux, fast=self.fast)


        if self.fast:
            results = [self._resample_and_convolve_loop((fiber, disp0, resampled_v0, tmp_lam, unit))
                       for fiber in self.bundle.fibers_science]
        else:
            with Pool(lvmdatasimulator.n_process) as pool:
                results = pool.map(self._resample_and_convolve_loop, [(fiber, disp0, resampled_v0,
                tmp_lam, unit) for fiber in self.bundle.fibers_science])

        out_spec = OrderedDict(results)

        return out_spec

    def _resample_and_convolve_loop(self, param): #fiber, disp0, resampled_v0, tmp_lam, unit):

        fiber = param[0]
        disp0 = param[1]
        resampled_v0 = param[2]
        tmp_lam = param[3]
        unit = param[4]

        fiber_spec = OrderedDict()
        for branch in self.spectrograph.branches:
            lsf_fwhm = branch.lsf_fwhm / disp0  # from A to pix

            convolved = convolve_for_gaussian(resampled_v0, lsf_fwhm, boundary="extend")
            resampled_v1 = resample_spectrum(branch.wavecoord.wave.value, tmp_lam, convolved,
                                             fast=self.fast)

            if unit:
                fiber_spec[branch.name] = resampled_v1 * unit
            else:
                fiber_spec[branch.name] = resampled_v1

        return fiber.id, fiber_spec

    def extract_target_spectra(self):
        """Extract spectra of the target from the field object"""

        wl_grid = np.arange(3500, 9910.01, 0.06) * u.AA

        log.info(f"Recovering target spectra for {self.bundle.nfibers} fibers.")
        index, spectra = self.source.extract_spectra(self.bundle.fibers_science, wl_grid,
                                                     obs_coords=self.observation.target_coords)

        log.info('Resampling spectra to the instrument wavelength solution.')

        if self.fast:
            results = [self._extract_target_spectra((fiber, spectra, index, wl_grid))
                       for fiber in self.bundle.fibers_science]
        else:
            with Pool(lvmdatasimulator.n_process) as pool:
                results = pool.map(self._extract_target_spectra, [(fiber, spectra, index, wl_grid)
                for fiber in self.bundle.fibers_science])

        obj_spec = OrderedDict(results)

        return obj_spec

    def _extract_target_spectra(self, param): #fiber, spectra, index, wl_grid):
        fiber = param[0]
        spectra = param[1]
        index = param[2]
        wl_grid = param[3]

        original = spectra[index == fiber.id, :][0]
        # from here, this is a replica of _resample_and_convolve()
        # I cannot use the method directly because I cannot use the same spectra for all fibers
        disp0 = np.median(wl_grid[1:-1] - wl_grid[0:-2])

        fiber_spec = OrderedDict()

        for branch in self.spectrograph.branches:
            lsf_fwhm = branch.lsf_fwhm / disp0  # from A to pix

            convolved = convolve_for_gaussian(original, lsf_fwhm, boundary="extend")
            resampled_v1 = resample_spectrum(branch.wavecoord.wave.value, wl_grid.value,
                                             convolved, fast=self.fast)

            fiber_spec[branch.name] = resampled_v1 * (u.erg / (u.cm ** 2 * u.s * u.AA))

        return fiber.id, fiber_spec


    def _simulate_observations_single_fiber(self, param): #fiber, spectra):
        """
        Simulate the observation of a single fiber.

        Args:
            fiber (Fiber):
                Fiber to be simulated
            spectra (dictionary):
                dictionary containing the input spectrum for each fiber.
        """

        fiber = param[0]
        spectra = param[1]
        exptime = param[2]

        spectrum = spectra[fiber.id]

        # convert spectra to electrons
        tmp_target = self._obj_to_electrons(spectrum, fiber.id, exptime)  # from units to electrons
        tmp_sky = self._sky_to_electrons(self.sky[fiber.id], exptime)

        # create 2D spectra
        tmp_target_2d = self._to_2d(fiber, tmp_target)
        tmp_sky_2d = self._to_2d(fiber, tmp_sky)
        tmp_noise_2d = self._make_noise(tmp_target_2d, tmp_sky_2d, exptime)

        # extract apertures
        single_exposure = self._extract_aperture(tmp_target_2d, tmp_noise_2d, tmp_sky_2d)

        # create a realistic spectrum with noise
        realization_noise = self._add_noise(single_exposure)
        # self.output_noise[fiber.id] = realization_noise

        # flux calibrate
        calibrated = self._flux_calibration(fiber.id, realization_noise, exptime)
        # self.output_calib[fiber.id] = calibrated

        return (fiber.id, realization_noise, calibrated, single_exposure)

    def simulate_observations(self, exptimes=None):
        """
        Runs the simulation, parallelizing it of the number of fibers to observe is large enough
        """

        log.info("Simulating observations.")
        self.extinction = self.extract_extinction()
        self.sky = self.extract_sky()
        self.target_spectra = self.extract_target_spectra()

        if exptimes is None:
            exptimes = self.observation.exptimes
        else:
            if not isinstance(exptimes, list):
                exptimes = [exptimes]
            log.warning('New exposure times have been provided. Overwriting the ones included in '
                        + 'Observation')

        for exptime in exptimes:
            exptime_unit = exptime * u.s
            if self.fast:
                results = [self._simulate_observations_single_fiber((fiber, self.target_spectra,
                                                                     exptime_unit))
                        for fiber in self.bundle.fibers_science]
            else:
                with Pool(lvmdatasimulator.n_process) as pool:
                    results = pool.map(self._simulate_observations_single_fiber, [(fiber,
                                                                                   self.target_spectra,
                                                                                   exptime_unit)
                    for fiber in self.bundle.fibers_science])

            # reorganize outputs
            ids = []  # fiber ids
            realizations = []  # realization without noise
            noises = []  # realization with noise
            calibs = []  # realization calibrated

            for item in results:
                ids.append(item[0])
                noises.append(item[1])
                calibs.append(item[2])
                realizations.append(item[3])

            self.output_no_noise[exptime] = OrderedDict(zip(ids, realizations))
            self.output_noise[exptime] = OrderedDict(zip(ids, noises))
            self.output_calib[exptime] = OrderedDict(zip(ids, calibs))

    def simulate_observations_custom_spectrum(self, wave, flux, norm=1, unit_wave=u.AA,
                                              unit_flux=u.erg*u.s**-1*u.cm**-2*u.arcsec**-2*u.AA**-1,
                                              exptimes=None):
        """
        Function to simulate the observations of a specific spectrum provided by the user.

        This function takes as an input a spectrum and it processes it in order to make it
        observable by the simulator. It then proceed simulating the observation of a single fiber.
        This function is supposed to be used only by the ETC.

        Args:
            exptimes (_type_, optional): _description_. Defaults to None.

        """

        log.info('Simulating real spectrum observation')

        # fixing the unit of measurement

        default_unit_wave = u.AA
        default_unit_flux = u.erg*u.s**-1*u.cm**-2*u.arcsec**-2*u.AA**-1

        if isinstance(wave, np.ndarray):
            wave *= unit_wave

        flux *= norm  # applying the normalization to the flux


        if not isinstance(flux, u.Quantity):
            flux *= unit_flux

        try:
            flux = flux.to(default_unit_flux)
        except u.UnitConversionError:
            raise u.UnitConversionError(f'Flux units {unit_flux} cannot be converted to {default_unit_flux}.')

        try:
            wave = wave.to(default_unit_wave)
        except u.UnitConversionError:
            raise u.UnitConversionError(f'Wavelength units {unit_wave} cannot be converted to {default_unit_wave}.')

        out_spectrum = OrderedDict()

        for fiber in self.bundle.fibers_science:

            branch_spec = OrderedDict()
            for branch in self.spectrograph.branches:

                dlam = (wave[-1] - wave[0]) / len(wave)

                lsf_fwhm = branch.lsf_fwhm / dlam

                convolved = convolve_for_gaussian(flux.value, lsf_fwhm, boundary="extend")
                resampled_v1 = resample_spectrum(branch.wavecoord.wave.value, wave.value,
                                                 convolved, fast=self.fast)

                resampled_v1 *= unit_flux
                to_flux = resampled_v1 * np.pi * (fiber.diameter/2)**2  # from SB to flux
                branch_spec[branch.name] = to_flux

            out_spectrum[fiber.id] = branch_spec

        self.target_spectra = out_spectrum
        self.extinction = self.extract_extinction()
        self.sky = self.extract_sky()

        # copy of the second part of simulate observations
        if exptimes is None:
            exptimes = self.observation.exptimes
        else:
            if not isinstance(exptimes, list):
                exptimes = [exptimes]
            log.warning('New exposure times have been provided. Overwriting the ones included in '
                        + 'Observation')
        for exptime in exptimes:
            exptime_unit = exptime * u.s
            if self.fast:
                results = [self._simulate_observations_single_fiber((fiber, self.target_spectra,
                                                                     exptime_unit))
                        for fiber in self.bundle.fibers_science]
            else:
                with Pool(lvmdatasimulator.n_process) as pool:
                    results = pool.map(self._simulate_observations_single_fiber, [(fiber,
                                                                                   self.target_spectra,
                                                                                   exptime_unit)
                    for fiber in self.bundle.fibers_science])

            # reorganize outputs
            ids = []
            realizations = []
            noises = []
            calibs = []
            for item in results:
                ids.append(item[0])
                noises.append(item[1])
                calibs.append(item[2])
                realizations.append(item[3])

            self.output_no_noise[exptime] = OrderedDict(zip(ids, realizations))
            self.output_noise[exptime] = OrderedDict(zip(ids, noises))
            self.output_calib[exptime] = OrderedDict(zip(ids, calibs))

    def save_outputs(self):
        """
        Main function to save the output of the simulation into rss file. Each different output is
        saved separately.
        """

        log.info("Saving the outputs:")
        for branch in self.spectrograph.branches:
            log.info('Input spectra')
            self._save_inputs(branch)
            log.info('Clean outputs')
            self._save_outputs_no_noise(branch)
            log.info('Uncalibrated outputs')
            self._save_outputs_with_noise(branch)
            log.info('Calibrated outputs')
            self._save_outputs_flux(branch)

    def _save_inputs(self, branch):
        """
        Save the input spectra for a certain branch to an RSS file.

        Args:
            branch (Branch):
                specific spectrograph branch to be saved.
        """

        ids, target, sky = self._reorganize_to_rss_input(branch)

        primary = self._create_primary_hdu(branch)

        signal_hdu = fits.ImageHDU(data=target.astype(np.float32), name="FLUX")
        signal_hdu.header["BUNIT"] = "erg/(cm2 s A)"
        primary.header["EXT1"] = "FLUX"

        sky_hdu = fits.ImageHDU(data=sky.astype(np.float32), name="SKY")
        sky_hdu.header["BUNIT"] = "erg/(cm2 s A)"
        primary.header["EXT2"] = "SKY"

        wave_hdu = fits.ImageHDU(data=branch.wavecoord.wave.value.astype(np.float32),
                                 name="WAVE")
        wave_hdu.header["BUNIT"] = "Angstrom"
        primary.header["EXT3"] = "WAVE"

        ids_hdu = fits.BinTableHDU(ids, name="FIBERID")
        primary.header["EXT4"] = "FIBERID"

        wcs_hdu = fits.ImageHDU(header=self._recover_wcs(), name='WCS')
        primary.header["EXT5"] = "WCS"

        hdul = fits.HDUList([primary, signal_hdu, sky_hdu, wave_hdu, ids_hdu, wcs_hdu])

        filename = os.path.join(self.outdir,
                                f"{self.source.name}_{branch.name}_{self.bundle.bundle_name}"
                                + "_input.fits")

        hdul.writeto(filename, overwrite=True)
        log.info(f"{filename} saved.")

    def _save_outputs_flux(self, branch):
        """
        Save the output flux calibrated spectra for a certain branch to an RSS file.

        Args:
            branch (Branch):
                specific spectrograph branch to be saved.
        """

        for exptime in self.observation.exptimes:
            ids, target, total, noise, sky, snr = self._reorganize_to_rss(branch,
                                                                          self.output_calib[exptime])

            primary = self._create_primary_hdu(branch, exptime)

            target_hdu = fits.ImageHDU(data=target.astype(np.float32), name="TARGET")
            target_hdu.header["BUNIT"] = "erg/ (cm2 s A)"
            primary.header["EXT1"] = "TARGET"

            total_hdu = fits.ImageHDU(data=total.astype(np.float32), name="TOTAL")
            total_hdu.header["BUNIT"] = "erg/ (cm2 s A)"
            primary.header["EXT2"] = "TOTAL"

            noise_hdu = fits.ImageHDU(data=noise.astype(np.float32), name="ERR")
            noise_hdu.header["BUNIT"] = "erg/ (cm2 s A)"
            primary.header["EXT3"] = "ERR"

            stn_hdu = fits.ImageHDU(data=snr.astype(np.float32), name="SNR")
            stn_hdu.header["BUNIT"] = ""
            primary.header["EXT4"] = "SNR"

            sky_hdu = fits.ImageHDU(data=sky.astype(np.float32), name="SKY")
            sky_hdu.header["BUNIT"] = "erg/ (cm2 s A)"
            primary.header["EXT5"] = "SKY"

            wave_hdu = fits.ImageHDU(data=branch.wavecoord.wave.value.astype(np.float32),
                                     name="WAVE")

            wave_hdu.header["BUNIT"] = "Angstrom"
            primary.header["EXT6"] = "WAVE"

            ids_hdu = fits.BinTableHDU(ids, name="FIBERID")
            primary.header["EXT7"] = "FIBERID"

            wcs_hdu = fits.ImageHDU(header=self._recover_wcs(), name='WCS')
            primary.header["EXT8"] = "WCS"

            hdul = fits.HDUList([primary, target_hdu, total_hdu, noise_hdu, stn_hdu, sky_hdu, wave_hdu,
                                ids_hdu, wcs_hdu])

            filename = os.path.join(self.outdir,
                                    f"{self.source.name}_{branch.name}_{self.bundle.bundle_name}_"
                                    + f"{exptime}_flux.fits")

            hdul.writeto(filename, overwrite=True)
            log.info(f"{filename} saved.")

    def _save_outputs_with_noise(self, branch):
        """
        Save the uncalibrated output spectra for a certain branch to an RSS file.

        Args:
            branch (Branch):
                specific spectrograph branch to be saved.
        """
        for exptime in self.observation.exptimes:
            ids, target, total, noise, sky, snr = self._reorganize_to_rss(branch,
                                                      self.output_noise[exptime])
            primary = self._create_primary_hdu(branch, exptime)

            target_hdu = fits.ImageHDU(data=target.astype(np.float32), name="TARGET")
            target_hdu.header["BUNIT"] = "e/pix"
            primary.header["EXT1"] = "TARGET"

            total_hdu = fits.ImageHDU(data=total.astype(np.float32), name="TOTAL")
            total_hdu.header["BUNIT"] = "e/pix"
            primary.header["EXT2"] = "TOTAL"

            noise_hdu = fits.ImageHDU(data=noise.astype(np.float32), name="ERR")
            noise_hdu.header["BUNIT"] = "e/pix"
            primary.header["EXT3"] = "ERR"

            stn_hdu = fits.ImageHDU(data=snr.astype(np.float32), name="SNR")
            stn_hdu.header["BUNIT"] = ""
            primary.header["EXT4"] = "SNR"

            sky_hdu = fits.ImageHDU(data=sky.astype(np.float32), name="SKY")
            sky_hdu.header["BUNIT"] = "e/pix"
            primary.header["EXT5"] = "SKY"

            wave_hdu = fits.ImageHDU(data=branch.wavecoord.wave.value.astype(np.float32),
                                     name="WAVE")
            wave_hdu.header["BUNIT"] = "Angstrom"
            primary.header["EXT6"] = "WAVE"

            ids_hdu = fits.BinTableHDU(ids, name="FIBERID")
            primary.header["EXT7"] = "FIBERID"

            wcs_hdu = fits.ImageHDU(header=self._recover_wcs(), name='WCS')
            primary.header["EXT8"] = "WCS"

            hdul = fits.HDUList([primary, target_hdu, total_hdu, noise_hdu, stn_hdu, sky_hdu, wave_hdu,
                                ids_hdu, wcs_hdu])

            filename = os.path.join(self.outdir,
                                    f"{self.source.name}_{branch.name}_{self.bundle.bundle_name}_"
                                    + f"{exptime}_realization.fits")

            hdul.writeto(filename, overwrite=True)
            log.info(f"{filename} saved.")

    def _save_outputs_no_noise(self, branch):
        """
        Save the uncalibrated output spectra without noise for a certain branch
        to an RSS file.

        Args:
            branch (Branch):
                specific spectrograph branch to be saved.
        """
        for exptime in self.observation.exptimes:
            ids, target, total, noise, sky, snr = self._reorganize_to_rss(branch,
                                                      self.output_no_noise[exptime])
            primary = self._create_primary_hdu(branch, exptime)

            target_hdu = fits.ImageHDU(data=target.astype(np.float32), name="TARGET")
            target_hdu.header["BUNIT"] = "e/pix"
            primary.header["EXT1"] = "TARGET"

            total_hdu = fits.ImageHDU(data=total.astype(np.float32), name="TOTAL")
            total_hdu.header["BUNIT"] = "e/pix"
            primary.header["EXT2"] = "TOTAL"

            noise_hdu = fits.ImageHDU(data=noise.astype(np.float32), name="ERR")
            noise_hdu.header["BUNIT"] = "e/pix"
            primary.header["EXT3"] = "ERR"

            stn_hdu = fits.ImageHDU(data=snr.astype(np.float32), name="SNR")
            stn_hdu.header["BUNIT"] = ""
            primary.header["EXT4"] = "SNR"

            sky_hdu = fits.ImageHDU(data=sky.astype(np.float32), name="SKY")
            sky_hdu.header["BUNIT"] = "e/pix"
            primary.header["EXT5"] = "SKY"

            wave_hdu = fits.ImageHDU(data=branch.wavecoord.wave.value.astype(np.float32),
                                     name="WAVE")
            wave_hdu.header["BUNIT"] = "Angstrom"
            primary.header["EXT6"] = "WAVE"

            ids_hdu = fits.BinTableHDU(ids, name="FIBERID")
            primary.header["EXT7"] = "FIBERID"

            wcs_hdu = fits.ImageHDU(header=self._recover_wcs(), name='WCS')
            primary.header["EXT8"] = "WCS"


            hdul = fits.HDUList([primary, target_hdu, total_hdu, noise_hdu, stn_hdu, sky_hdu, wave_hdu,
                                ids_hdu, wcs_hdu])

            filename = os.path.join(self.outdir,
                                    f"{self.source.name}_{branch.name}_{self.bundle.bundle_name}_"
                                    + f"{exptime}_no_noise.fits")

            hdul.writeto(filename, overwrite=True)
            log.info(f"{filename} saved.")

    def _obj_to_electrons(self, spectrum, fiber_id, exptime):
        """
        Convert a spectrum in flux units to electrons per second.

        Args:
            spectrum (OrderedDict):
                Dictionary containing the spectrum in all the simulated branches
            fiber_id (int):
                id of the selected fiber

        Returns:
            OrderedDict:
                Dictionary containing the spectra in the observed branches transformed to counts/s
        """

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
            constant = exptime * branch.efficiency() *\
                self.telescope.aperture_area

            # atmospheric extinction
            atmosphere = self.extinction[fiber_id][branch.name] * self.observation.airmass

            # put everything together
            out[branch.name] = spectrum_e * constant * 10 ** (-0.4 * (atmosphere)) * u.electron

        return out

    def _to_2d(self, fiber, spectrum):
        """
        This function transform the 1D spectrum in a 2D spectrum, to simulate the dispersion
        on the detector caused by the fiber.
        """

        nypix = int(round_up_to_odd(fiber.nypix.value))

        out = OrderedDict()
        for branch in self.spectrograph.branches:
            new_2d_array = np.zeros((len(spectrum[branch.name]), nypix), dtype=np.float32)
            ne_frac = np.zeros(nypix, dtype=np.float32)
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

    def _sky_to_electrons(self, sky_spectrum, exptime):
        """
        Convert the sky spectrum from flux units to counts per second.

        Args:
            sky_spectrum (OrderedDict):
                Input sky spectrum.

        Returns:
            OrderedDict:
                Dictionary containing the transformed spectrum
        """

        out = OrderedDict()
        for branch in self.spectrograph.branches:
            # convert spectrum to electrons
            sky_e = flam2epp(branch.wavecoord.wave, sky_spectrum[branch.name],
                             branch.wavecoord.step)

            # compute constant
            constant = exptime * branch.efficiency() * self.telescope.aperture_area
            out[branch.name] = sky_e * constant * u.electron

        return out

    def _make_noise(self, spectrum, sky, exptime):
        """
        Generate the error array by combining all the different sources of error. Gaussian errors
        are considered.

        Args:
            spectrum (OrderedDict):
                Input spectrum
            sky (OrderedDict):
                Input sky spectrum

        Returns:
            OrderedDict:
                Dictionary containing the error arrays, one for each simulated branch
        """

        out = OrderedDict()

        # this is to create more realistic noise
        # ron = np.random.randn(spectrum[branch.name].shape) * branch.ron

        for branch in self.spectrograph.branches:
            dark = branch.dark * exptime
            out[branch.name] = np.sqrt(spectrum[branch.name].value + sky[branch.name].value +
                                       branch.ron.value ** 2 + dark.value) * \
                spectrum[branch.name].unit

        return out

    def _extract_aperture(self, spec2d, noise2d, sky2d):
        """
        Extract a 1D spectrum from the simulated 2D realizations

        Args:
            spec2d (OrderedDict):
                Dictionary containing the 2D spectra to be extracted.
            noise2d (OrderedDict):
                Dictionary containing the 2D error arrays to be extracted.
            sky2d (OrderedDict):
                Dictionary containing the 2D spectra of the sky to be extracted.


        Returns:
            _type_: _description_
        """

        signal_to_noise_out = OrderedDict()
        target_out = OrderedDict()
        noise_out = OrderedDict()
        sky_out = OrderedDict()
        total_out = OrderedDict()

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
            flux_auxiliar = np.zeros(sely.shape, dtype=np.float32)
            error_auxiliar = np.zeros(sely.shape, dtype=np.float32)
            sky_auxiliar = np.zeros(sely.shape, dtype=np.float32)
            flux_auxiliar[sely] = spec2d[branch.name][sely]  # spectrum
            error_auxiliar[sely] = noise2d[branch.name][sely]  # noise
            sky_auxiliar[sely] = sky2d[branch.name][sely]  # sky
            target = flux_auxiliar.sum(axis=1)
            noise = np.sqrt(np.sum(error_auxiliar ** 2, axis=1))
            sky = sky_auxiliar.sum(axis=1)
            signal_to_noise_out[branch.name] = target / noise
            target_out[branch.name] = target
            noise_out[branch.name] = noise
            sky_out[branch.name] = sky
            total_out[branch.name] = sky + target

        return {"signal": total_out, "target": target_out, "noise": noise_out,
                "snr": signal_to_noise_out, "sky": sky_out}

    def _add_noise(self, exposure):
        """
        Add a gaussian noise to the observed spectra. The noise is added generating an array
        where each pixel is randomly extracted from a Gaussian distribution with standard
        deviation equal to the value of the pixel in the error array.

        Args:
            exposure (OrderedDict):
                Dictionary containing the spectra and the error array

        Returns:
            Dict:
                Dictionary containing the observed spectrum with noise, the target spectrum with
                noise, the noise array, the sky spectrum with noise, the signal to noise spectrum
        """

        signal_noise = OrderedDict()
        obj_noise = OrderedDict()
        sky_noise = OrderedDict()

        for branch in self.spectrograph.branches:

            noise_to_add = np.random.standard_normal(exposure["noise"][branch.name].shape) *\
                exposure["noise"][branch.name]

            signal_with_noise = exposure["signal"][branch.name] + noise_to_add
            target = signal_with_noise - exposure["sky"][branch.name]
            sky = signal_with_noise - exposure["target"][branch.name]
            signal_noise[branch.name] = signal_with_noise
            obj_noise[branch.name] = target
            sky_noise[branch.name] = sky

        return {
            "signal": signal_noise,
            "target": obj_noise,
            "noise": exposure["noise"],
            "sky": sky_noise,
            "snr": exposure["snr"],
        }

    def _coadd(self, spectrum, sky, noise):
        """
        Not Used yet

        Args:
            spectrum (_type_): _description_
            sky (_type_): _description_
            noise (_type_): _description_

        Returns:
            _type_: _description_
        """

        spec_coadd = OrderedDict()
        sky_coadd = OrderedDict()
        noise_coadd = OrderedDict()

        for branch in self.spectrograph.branches:

            spec_coadd[branch.name] = spectrum[branch.name] * self.observation.nexp
            sky_coadd[branch.name] = sky[branch.name] * self.observation.nexp
            noise_coadd[branch.name] = noise[branch.name] * np.sqrt(self.observation.nexp)

        return spec_coadd, sky_coadd, noise_coadd

    def _reorganize_to_rss_input(self, branch):
        """
        Reorganize the input spectra to be saved to an RSS file. Also a table connecting the
        spectra to the fiber is created.

        Args:
            branch (Branch):
                Branche to be saved.

        Returns:
            astropy.table.Table:
                Table containing the basic information on the fibers included in the bundle
            numpy.array:
                array containing the rearranged observed spectra (target+sky)
            numpy.array:
                array containing the rearranged target spectra
        """

        nfibers = self.bundle.nfibers
        fib_id = []

        signal = np.zeros((nfibers, branch.wavecoord.npix), dtype=np.float32)
        sky = np.zeros((nfibers, branch.wavecoord.npix), dtype=np.float32)

        for i, fiber in enumerate(self.bundle.fibers_science):
            fib_id.append(fiber.to_table())

            signal[i, :] = self.target_spectra[fiber.id][branch.name]
            sky[i, :] = self.sky[fiber.id][branch.name]

        fib_id = vstack(fib_id)
        return fib_id, signal, sky

    def _reorganize_to_rss(self, branch, exposures):
        """
        Reorganize the output spectra to be saved to an RSS file. Also a table connecting the
        spectra to the fiber is created. It can rearrange both the uncalibrated and the
        calibrated spectra.

        Args:
            branch (Branch):
                Only the spectra of this branch will be rearranged
            exposures (dict):
                Dictionary containing the spectra to be rearranged.

        Returns:
            astropy.table.Table:
                Table containing the basic information on the fibers included in the bundle
            numpy.array:
                array containing the rearranged target spectra
            numpy.array:
                array containing the rearranged observed spectra (target+sky)
            numpy.array:
                array containing the rearranged noise spectra
            numpy.array:
                array containing the rearranged observed spectra (target+sky)
            numpy.array:
                array containing the rearranged target spectra

        """

        nfibers = self.bundle.nfibers
        fib_id = []

        # outputs with noise
        target = np.zeros((nfibers, branch.wavecoord.npix), dtype=np.float32)
        total = np.zeros((nfibers, branch.wavecoord.npix), dtype=np.float32)
        sky = np.zeros((nfibers, branch.wavecoord.npix), dtype=np.float32)
        noise = np.zeros((nfibers, branch.wavecoord.npix), dtype=np.float32)
        snr = np.zeros((nfibers, branch.wavecoord.npix), dtype=np.float32)

        for i, spectra in enumerate(exposures.values()):
            fib_id.append(self.bundle.fibers_science[i].to_table())
            target[i, :] = spectra["target"][branch.name]
            total[i, :] = spectra["signal"][branch.name]
            noise[i, :] = spectra["noise"][branch.name]
            sky[i, :] = spectra["sky"][branch.name]
            snr[i, :] = spectra["snr"][branch.name]

        fib_id = vstack(fib_id)
        return fib_id, target, total, noise, sky, snr

    def _create_primary_hdu(self, branch, exptime=None):

        primary = fits.PrimaryHDU()
        primary.header["TARGET"] = self.source.name
        primary.header["RA"] = (self.source.ra.value, 'ra of the observed field')
        primary.header["DEC"] = (self.source.dec.value, 'dec of the observed field')
        primary.header["OBS_RA"] = (self.observation.ra.to(u.deg).value, 'ra of the fiber array')
        primary.header["OBS_DEC"] = (self.observation.dec.to(u.deg).value, 'dec of the fiber array')
        primary.header["AZ"] = (self.observation.target_coords_altaz.az.value,
                                "Azimuth of the fiber array")
        primary.header["ALT"] = (self.observation.target_coords_altaz.alt.value,
                                 "Altitude of the fiber target")
        primary.header["AIRMASS"] = self.observation.airmass
        primary.header["MJD"] = (self.observation.mjd, "MJD at start")
        if exptime is not None:
            primary.header["EXPTIME"] = exptime
        primary.header["BRANCH"] = branch.name
        primary.header["MOON"] = (self.observation.moon_distance.value,
                                  "Fractional moon illumination")
        primary.header["DAY-MOON"] = (self.observation.days_moon,
                                      "Days from new moon")

        primary.header['SIMUL'] = ('1D', 'Simulator version (1D or 2D)')

        return primary

    def _flux_calibration(self, fiber_id, exposure, exptime):

        signal_out = OrderedDict()
        target_out = OrderedDict()
        sky_out = OrderedDict()
        noise_out = OrderedDict()

        for branch in self.spectrograph.branches:
            # this remove the signature of the instruments and goes back to the real spectrum
            fluxcalib = exptime * branch.efficiency() *\
                self.telescope.aperture_area
            telluric = self.extinction[fiber_id][branch.name] * self.observation.airmass

            # apply flux calibration
            tmp_signal = exposure['signal'][branch.name] / fluxcalib
            tmp_sky = exposure['sky'][branch.name] / fluxcalib
            tmp_noise = exposure['noise'][branch.name] / fluxcalib
            tmp_target = exposure['target'][branch.name] / fluxcalib

            # apply telluric calibration
            tmp_noise /= 10 ** (-0.4 * telluric)
            tmp_target /= 10 ** (-0.4 * telluric)

            signal_out[branch.name] = epp2flam(branch.wavecoord.wave, tmp_signal,
                                               branch.wavecoord.step)
            target_out[branch.name] = epp2flam(branch.wavecoord.wave, tmp_target,
                                               branch.wavecoord.step)
            sky_out[branch.name] = epp2flam(branch.wavecoord.wave, tmp_sky,
                                            branch.wavecoord.step)
            noise_out[branch.name] = epp2flam(branch.wavecoord.wave, tmp_noise,
                                              branch.wavecoord.step)

        return {"signal": signal_out,
                "target": target_out,
                "noise": noise_out,
                "snr": exposure['snr'],
                "sky": sky_out}

    def save_output_maps(self, wavelength_ranges, unit_range=u.AA):
        """
        Save 2D images of the observed field in all the provided wavelength ranges.
        To better show the results of the simulation, the flux observed by each fiber is reported
        inside a circle with the same diameter of the fiber.
        For this reason, these images are more a qualitative way to visualize the results of the
        observations and not a scientific product.

        Args:
            wavelength_ranges (list):
                Wavelength ranges to extract from the simulated data. Each wavelength range should
                be provided as a list, e.g. [[wave1_0, wave1_1], [wave2_0, wave2_1]]. If only one
                range should be extracted, it can be provided as a single list (i.e. [wave0, wave1])
            unit_range (astropy.unit, optional):
                wavelength units used to define the wavelength ranges. All ranges should be
                defined using the same units. Defaults to u.AA.
        """

        log.info('Saving the 2D output maps')

        if isinstance(wavelength_ranges[0], (float, int)):
            wavelength_ranges = [wavelength_ranges]

        # applying the units of measurement and converting to angstrom
        for wavelength_range in wavelength_ranges:
            wavelength_range[0] = wavelength_range[0] * unit_range
            wavelength_range[0] = wavelength_range[0].to(u.AA).value
            wavelength_range[1] = wavelength_range[1] * unit_range
            wavelength_range[1] = wavelength_range[1].to(u.AA).value

        for wavelength_range in wavelength_ranges:
            for branch in self.spectrograph.branches:
                if branch.wavecoord.start < wavelength_range[0] * unit_range \
                    and branch.wavecoord.end > wavelength_range[1] * unit_range:
                    for exptime in self.observation.exptimes:
                        ids, target, total, _, _, _ = self._reorganize_to_rss(branch,
                                                                              self.output_calib[exptime])
                        target_out = np.zeros((self.source.npixels, self.source.npixels),
                                              dtype=np.float32)
                        total_out = np.zeros((self.source.npixels, self.source.npixels),
                                              dtype=np.float32)

                        wcs = self.source.wcs
                        head = wcs.to_header()

                        head['MIN_WAVE'] = wavelength_range[0]
                        head['MAX_WAVE'] = wavelength_range[1]

                        # I'm not interpolating to the exact wavelength

                        mask1 = branch.wavecoord.wave > wavelength_range[0] * unit_range
                        mask2 = branch.wavecoord.wave < wavelength_range[1] * unit_range
                        mask = np.all([mask1, mask2], axis=0)

                        dl = branch.wavecoord.step.value

                        target_val = np.nansum(target[:, mask], axis=1) * dl
                        total_val = np.nansum(total[:, mask], axis=1) * dl

                        # Just the target
                        target_out = self._populate_map(target_out, target_val, ids, wcs)

                        filename = os.path.join(self.outdir,
                            f"{self.source.name}_{branch.name}_{self.bundle.bundle_name}"\
                            + f"_{int(wavelength_range[0])}_{int(wavelength_range[1])}"
                            + f"_{exptime}s_target_map.fits")

                        hdu = fits.PrimaryHDU(data=target_out.astype(np.float32), header=head)

                        hdu.writeto(filename, overwrite=True)
                        log.info(f' Saving {filename}...')

                        # full spectrum
                        total_out = self._populate_map(total_out, total_val, ids, wcs)

                        filename = os.path.join(self.outdir,
                            f"{self.source.name}_{branch.name}_{self.bundle.bundle_name}"
                            + f"_{int(wavelength_range[0])}_{int(wavelength_range[1])}"
                            + f"_{exptime}s_total_map.fits")
                        hdu = fits.PrimaryHDU(data=total_out.astype(np.float32), header=head)

                        hdu.writeto(filename, overwrite=True)
                        log.info(f' Saving {filename}...')

                else:
                    log.warning(f'Selected range out of {branch.name} range.')

        self._print_fibers_to_ds9_regions()

    def _print_fibers_to_ds9_regions(self):

        outname = os.path.join(self.outdir, f'{self.source.name}_fibers.reg')

        with open(outname, 'w') as f:
            print('# Region file format: DS9 version 4.1', file=f)
            print('global color=green dashlist=8 3 width=1 font="helvetica 10 normal roman"',
                  end=' ', file=f)
            print('select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1',
                  file=f)
            print('image', file=f)

            for fiber in self.bundle.fibers_science:
                # converting to pixels
                coord = self.observation.target_coords.spherical_offsets_by(fiber.x, fiber.y)
                x, y = self.source.wcs.all_world2pix(coord.ra, coord.dec, 1)
                r = fiber.diameter / (2 * self.source.pxsize.to(u.arcsec))
                print(f'circle({x:0.3f}, {y:0.3f}, {r:0.3f}) # text={{{fiber.id}}}', file=f)

        log.info(f' Saving {outname}...')

    def _populate_map(self, map, values, ids, wcs):

        # I'm assuming that all fibers will have the same diameter on the sky
        # For now this is fine, but it might not be ok anymore with the real instrument
        diameter = np.ceil(self.bundle.fibers_science[0].diameter / self.source.pxsize).value
        if diameter % 2 == 0:
            size = int(diameter) + 3
        else:
            size = int(diameter) + 2

        kernel = np.zeros((size, size), dtype=np.float32)
        center = kernel.shape[0] // 2  # center of new array

        yy, xx = np.mgrid[:kernel.shape[0], :kernel.shape[1]]
        radius = np.sqrt((xx - center)**2 + (yy - center)**2)
        kernel[radius < diameter / 2] = 1

        for fiber in self.bundle.fibers_science:
            flux = values[ids['id'] == fiber.id]
            fiber_coord = self.source.coord.spherical_offsets_by(fiber.x, fiber.y)

            fiber_x, fiber_y = wcs.world_to_pixel(fiber_coord)
            fiber_x = int(np.rint(fiber_x))
            fiber_y = int(np.rint(fiber_y))

            map[fiber_y-center: fiber_y+center+1, fiber_x-center: fiber_x+center+1] += kernel * flux

        return map

    def _recover_wcs(self):

        header = self.source.wcs.to_header()

        header['OBS_RA'] = (self.observation.ra.to(u.deg).value, 'ra of the fiber array')
        header['OBS_DEC'] = (self.observation.dec.to(u.deg).value, 'dec of the fiber array')

        return header