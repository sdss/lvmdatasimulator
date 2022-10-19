import numpy as np
import matplotlib.pyplot as plt
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

from lvmdatasimulator import DATA_DIR, WORK_DIR
from lvmdatasimulator.instrument import Spectrograph
from lvmdatasimulator.field import LVMField
from lvmdatasimulator.fibers import FiberBundle
from lvmdatasimulator.observation import Observation
from lvmdatasimulator.telescope import Telescope
from lvmdatasimulator.simulator import flam2epp, resample_spectrum
from lvmdatasimulator.stars import StarsList
from lvmdatasimulator import log
from lvmdatasimulator.utils import round_up_to_odd, set_geocoronal_ha, open_sky_file
from joblib import Parallel, delayed

import os
import sys
import imp
twodlvm = imp.load_source('2d_LVM', f'{DATA_DIR}/../../LVM_2D/2d_projection.py')


def reduce_size(spectrum, wave, wave_min, wave_max):

    mask = np.all([wave > wave_min, wave< wave_max], axis=0)
    out_shape = (spectrum.shape[0], mask.sum())
    if len(spectrum.shape) == 2:
        mask = expand_to_full_fiber(mask, spectrum.shape[0])
        out = spectrum[mask].reshape(out_shape)
    else:
        out = spectrum[mask]

    return out


def expand_to_full_fiber(input_array, nfibers):

    if len(input_array.shape) > 2:
        raise ValueError(f'input_array should be a 1d or 2d array, but it is {len(input_array.shape)}')

    input_array = np.atleast_2d(input_array)
    output = np.repeat(input_array, nfibers, axis=0)

    return output


def get_fibers_table(science=None):
    """Build the table that tells which fibers are being used"""

    sky1 = ascii.read(os.path.join(DATA_DIR, 'instrument', 'sky1_array.dat'))
    sky2 = ascii.read(os.path.join(DATA_DIR, 'instrument', 'sky2_array.dat'))
    std = ascii.read(os.path.join(DATA_DIR, 'instrument', 'std_array.dat'))

    if science is None:
        science = ascii.read(os.path.join(DATA_DIR, 'instrument', 'full_array.dat'))

    new = vstack([science, sky1, sky2, std])

    ringid = new['ring_id']
    fibtype = new['type']
    pos = new['fiber_id']
    fibid = np.arange(len(new), dtype=int)

    return ringid, pos, fibtype, fibid


class Simulator2D:
    def __init__(
        self,
        source: LVMField,
        observation: Observation,
        spectrograph: Spectrograph,
        bundle: FiberBundle,
        telescope: Telescope,
        aperture: u.pix = 10 * u.pix,
        root: str = WORK_DIR,
        overwrite: bool = True,
        fast: bool = True,
    ):

        self.source = source
        self.observation = observation
        self.spectrograph = spectrograph
        self.bundle = bundle
        self.telescope = telescope
        self.aperture = aperture
        self.root = root
        self.overwrite = overwrite
        self.fast = fast

        # creating empty storage
        self.output = None
        # self.output_coadd = OrderedDict()

        self.outdir = os.path.join(self.root, "outputs")
        if os.path.isdir(self.outdir) and not self.overwrite:
            log.warning(f"{self.outdir} already exist. Terminating the simulation.")
            return
        if not os.path.isdir(self.outdir):
            os.mkdir(self.outdir)

        # data storage
        self.extinction = None
        self.sky = None
        self.target_spectra = None
        self.total_spectra = None
        self.standards = None
        self.index = None
        self.flat = None
        self.std = None
        self.arc = None

        # auxiliary definitions
        self._disp = 0.06 * u.AA/ u.pix
        self._wl_grid = np.arange(3500, 9910.01, self._disp.value) * u.AA
        self._area_fiber = np.pi * (self.bundle.fibers[0].diameter / 2) ** 2
        self._fibers_per_spec = int(self.bundle.max_fibers / 3)


    def create_flat(self):

        log.info('Creating flat fields...')

        filename = os.path.join(DATA_DIR, 'lamps', 'flat_sb.dat')
        data = ascii.read(filename)

        flat = resample_spectrum(self._wl_grid, data['wave'], data['flux'], fast=self.fast)

        self._project_2d_calibs(flat, 'flat')


    def create_arc(self, hg=False, ne=False, ar=False, xe=False):

        if not np.any([hg, ne, ar, xe]):
            raise ValueError('At least one lamp should be selected')

        arcs = np.zeros((4, len(self._wl_grid)))

        if hg:
            filename = os.path.join(DATA_DIR, 'lamps', 'mercury_arc_sb.dat')
            data = ascii.read(filename)

            log.info('Including Hg lamp...')

            arcs[0] = resample_spectrum(self._wl_grid, data['wave'], data['flux'],fast=self.fast)

        if ne:
            filename = os.path.join(DATA_DIR, 'lamps', 'neon_arc_sb.dat')
            data = ascii.read(filename)

            log.info('Including Ne lamp...')

            arcs[1] = resample_spectrum(self._wl_grid, data['wave'], data['flux'],fast=self.fast)

        if ar:
            filename = os.path.join(DATA_DIR, 'lamps', 'argon_arc_sb.dat')
            data = ascii.read(filename)

            log.info('Including Ar lamp...')

            arcs[2] = resample_spectrum(self._wl_grid, data['wave'], data['flux'],fast=self.fast)

        if xe:
            filename = os.path.join(DATA_DIR, 'lamps', 'xenon_arc_sb.dat')
            data = ascii.read(filename)

            log.info('Including Xe lamp...')

            arcs[3] = resample_spectrum(self._wl_grid, data['wave'], data['flux'],fast=self.fast)


        # collapsing in the single required arc
        arcs = arcs.sum(axis=0)


        self._project_2d_calibs(arcs, 'arc', self.observation.arc_exptimes)

    def extract_extinction(self, extinction_file=os.path.join(DATA_DIR, 'sky',
                                                              'LVM_LVM160_KLAM.dat')):

        log.info('Reading the atmospheric extinction from file.')
        self.extinction_file = extinction_file
        data = ascii.read(self.extinction_file)

        self.extinction = resample_spectrum(self._wl_grid, data['col1'], data['col2'],
                                            fast=self.fast)

    def extract_sky(self, unit=u.erg / (u.cm ** 2 * u.s * u.AA)):
        """
        Return sky emission spectrum sampled at instrumental wavelengths
        """

        wave, brightness = open_sky_file(self.observation.sky_template, self.observation.days_moon,
                                         self.telescope.name)

        if self.observation.geocoronal is not None:
            ha = self.observation.geocoronal
            brightness = set_geocoronal_ha(wave, brightness, ha)

        flux = brightness * self._area_fiber.value  # converting to Fluxes from SBrightness

        log.info('Resample sky emission to instrument wavelength solution.')
        self.sky = resample_spectrum(self._wl_grid, wave, flux, fast=self.fast) * unit

    def extract_target_spectra(self, unit=u.erg / (u.cm ** 2 * u.s * u.AA)):

        log.info(f"Recovering target spectra for {self.bundle.nfibers} fibers.")
        index, spectra = self.source.extract_spectra(self.bundle.fibers, self._wl_grid,
                                                     obs_coords=self.observation.target_coords)
        self.index = index
        self.target_spectra = spectra

    def extract_std_spectra(self, nstd=24, tmin=5500, tmax=8000, dt=500,
                            gmin=8, gmax=12, dg=0.1):

        log.info('Generating standard stars')
        # selecting the temperature of the stars
        T_aval = np.arange(tmin, tmax+dt, dt, dtype=int)  # possible values
        T_sel = np.random.choice(T_aval, nstd)  # randomly selected

        # generating the brightness of these stars
        g_aval = np.arange(gmin, gmax+dg, dg)
        g_sel = np.random.choice(g_aval, nstd)


        # generate the star list
        stars = StarsList()
        for T, g in zip(T_sel, g_sel):
            stars.add_star(ra=0, dec=0, gmag=g, teff=T, ag=0, v=0, check=False)

        stars.associate_spectra(shift=False)
        stars.rescale_spectra()

        self.standards = stars

    def simulate_science(self):

        unit_e = u.electron / (u.s * u.pix * u.cm**2)
        # get the target spectra and expand the array to cover all the unused fibers
        # I assume that the fiber rearranging will be performed at a later stage
        self.extract_target_spectra()
        spectra = np.zeros_like(self.target_spectra.value) * unit_e
        for i, spectrum in enumerate(self.target_spectra):
            spectra[i] = flam2epp(self._wl_grid, spectrum, self._disp) * u.electron

        # simulating spectra
        spectra *= self.telescope.aperture_area
        self.extract_extinction()
        extinction = expand_to_full_fiber(self.extinction, len(spectra))
        atmosphere = self.observation.airmass * extinction
        spectra *= 10 ** (-0.4 * (atmosphere))

        #simulating sky
        self.extract_sky()
        sky_e = flam2epp(self._wl_grid, self.sky, self._disp) * u.electron
        sky_fiber = sky_e * self.telescope.aperture_area
        sky = expand_to_full_fiber(sky_fiber, len(self.index))

        assert sky.shape == spectra.shape, 'Something wrong with sky and spectra'

        self.total_spectra = sky + spectra   # this is in electron
        # science_projection = np.zeros((self.bundle.max_obj_fibers, len(self._wl_grid)))
        # for i, spec in enumerate(self.total_spectra):
        #     science_projection[i] = spec

        # reshape sky spectra
        sky_projection = expand_to_full_fiber(sky_fiber, self.bundle.max_sky_fibers)

        # get standard stars spectra
        self.extract_std_spectra()
        standards = np.zeros((len(self.standards), len(self._wl_grid))) * unit_e
        for i, std in enumerate(self.standards.spectra):
            tmp = resample_spectrum(self._wl_grid.value, self.standards.wave.value, std, fast=True)
            standards[i] = flam2epp(self._wl_grid,
                                    tmp*u.erg / (u.cm ** 2 * u.s * u.AA),
                                    self._disp) * u.electron
        standards *= self.telescope.aperture_area
        extinction = expand_to_full_fiber(self.extinction, len(standards))
        atmosphere = self.observation.airmass * extinction
        standards *= 10 ** (-0.4 * (atmosphere))

        sky_std = expand_to_full_fiber(sky_fiber, len(standards))

        standards += sky_std
        self._project_2d(self.total_spectra, sky_projection, standards)


    def _project_2d(self, science, sky, std):

        # get the tables for the different components of the final output
        sci_fibers = self.bundle.fibers_table

        ringid, pos, fibtype, fibid = get_fibers_table(science=sci_fibers)

        new_sky = {}
        new_sci = {}
        new_std = {}

        # separating the full spectrum in the different branches

        for branch in self.spectrograph.branches:
            tmp_sky = sky * branch.efficiency(self._wl_grid)
            tmp_science = science * branch.efficiency(self._wl_grid)
            tmp_std = std * branch.efficiency(self._wl_grid)

            new_sky[branch.name] = reduce_size(tmp_sky, self._wl_grid, branch.wavecoord.start,
                                               branch.wavecoord.end)
            new_sci[branch.name] = reduce_size(tmp_science, self._wl_grid, branch.wavecoord.start,
                                               branch.wavecoord.end)
            new_std[branch.name] = reduce_size(tmp_std, self._wl_grid, branch.wavecoord.start,
                                               branch.wavecoord.end)

        for time in self.observation.exptimes:
            for time_std in self.observation.std_exptimes:
                log.info(f'Saving science exposures with {time}s exposures and {time_std}s '
                         'of exposure for each standard star')

                for branch in self.spectrograph.branches:
                    name = branch.name

                    sci_corr = new_sci[name] * time
                    sky_corr = new_sky[name] * time
                    std_corr = new_std[name] * time_std

                    spectra_final = np.vstack([sci_corr, sky_corr, std_corr])

                    self._to_camera(spectra_final, fibid, fibtype, ringid, pos,
                                    time, name, 'science', branch)

    def _project_2d_calibs(self, data, calib_name, exptimes):

        # moving from SB to flux
        # the arc files are already in e/s/pix/arcsec^2 no need to tranform and multiply
        # per telescope area
        calib = data * self._area_fiber.value

        # for now the calibrations are independent from the position of the fibers in the field
        # THIS WILL CHANGE IN THE FUTURE.

        ringid, pos, fibtype, fibid = get_fibers_table(science=None)

        # applying the sensitivity function and reducing the size to spare memory
        new_calib = {}
        for branch in self.spectrograph.branches:
            tmp = calib * branch.efficiency(self._wl_grid)
            new_calib[branch.name] = reduce_size(tmp, self._wl_grid,
                                                 branch.wavecoord.start, branch.wavecoord.end)

        for time in exptimes:

            log.info(f'Saving {calib_name} exposures with exptime {time}s')
            for branch in self.spectrograph.branches:
                name = branch.name
                calib = new_calib[name] * time

                calib_final = expand_to_full_fiber(calib, self.bundle.max_fibers)
                # calib_final = calib_final.T

                self._to_camera(calib_final, fibid, fibtype, ringid, pos, self.observation.narcs,
                                name, calib_name, branch)

    def _to_camera(self, spectra, fibid, fibtype, ringid, pos, exptime, name, exp_type, branch):

        n_cr = int(branch.cosmic_rates.value * exptime)

        if name == 'blue':
            expn='00002998'
            cam='b1'
        elif name == 'red':
            expn='00001563'
            cam='r1'
        elif name == 'ir':
            expn='00001563'
            cam='z1'

        cube_file = cube_file=f'{DATA_DIR}/lab/sdR-s-{cam}-{expn}.disp.fits'
        wave2d, _ = fits.getdata(cube_file, 0, header=True)
        wave_s=np.nanmean(wave2d,axis=0)

        # this is a good point for parallelization but we need to modify run_2d
        for cam in range(0, 3):
            twodlvm.run_2d(spectra, fibid=fibid, fibtype=fibtype, ring=ringid,
                            position=pos, wave_s=wave_s, wave=self._wl_grid,
                            nfib=self._fibers_per_spec, type=name,
                            cam=cam+1, n_cr=n_cr, expN=expn,
                            expt=self.observation.narcs, ra=0, dec=0, mjd=str(self.observation.mjd),
                            flb=exp_type, base_name='sdR', dir1=self.outdir)

