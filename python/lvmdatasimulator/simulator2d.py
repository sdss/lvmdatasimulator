# encoding: utf-8
# @Author: Oleg Egorov, Enrico Congiu
# @Date: Oct 28, 2022
# @Filename: simulator2d.py
# @License: BSD 3-Clause
# @Copyright: Oleg Egorov, Enrico Congiu

import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u

from multiprocessing import Pool
from astropy.io import ascii, fits
from astropy.table import vstack


import lvmdatasimulator
from lvmdatasimulator import DATA_DIR, WORK_DIR
from lvmdatasimulator import COMMON_SETUP_2D as config_2d
from lvmdatasimulator.instrument import Spectrograph
from lvmdatasimulator.field import LVMField
from lvmdatasimulator.fibers import FiberBundle
from lvmdatasimulator.observation import Observation
from lvmdatasimulator.telescope import Telescope
from lvmdatasimulator.stars import StarsList
from lvmdatasimulator import log
import lvmdatasimulator.utils as util
from joblib import Parallel, delayed
import glob

import os
from lvmdatasimulator.projection2d import cre_raw_exp


def reduce_size(spectrum, wave, wave_min, wave_max, delta_w=0):

    mask = np.all([wave > (wave_min-delta_w), wave < (wave_max+delta_w)], axis=0)
    newwave = wave[mask]
    out_shape = (spectrum.shape[0], mask.sum())
    if len(spectrum.shape) == 2:
        mask = expand_to_full_fiber(mask, spectrum.shape[0])
        out = spectrum[mask].reshape(out_shape)
    else:
        out = spectrum[mask]

    return out, newwave


def expand_to_full_fiber(input_array, nfibers):
    """"""

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
        science = ascii.read(os.path.join(DATA_DIR, 'instrument', 'science_array.dat'))

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
        compress: bool = True,
        save_std: bool = True
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
        self.compress = compress
        self.save_std = save_std

        # use field coords as default if nothing is given in Observation
        if self.observation.target_coords is None:
            self.observation.target_coords = self.source.coord
            self.observation.ra = self.source.ra
            self.observation.dec = self.source.dec

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
        self.sky1 = None
        self.sky2 = None
        self.target_spectra = None
        self.total_spectra = None
        self.standards = None
        self.index = None
        self.flat = None
        self.std = None
        self.arc = None

        # auxiliary definitions
        self._disp = 0.06 * u.AA / u.pix
        self._wl_grid = np.arange(3500, 9910.01, self._disp.value) * u.AA
        self._area_fiber = np.pi * (self.bundle.fibers_science[0].diameter / 2) ** 2
        self._fibers_per_spec = int(self.bundle.max_fibers / 3)

    def create_flat(self, overwrite=True):

        log.info('Creating flat fields...')

        filename = os.path.join(DATA_DIR, 'lamps', 'flat.dat')
        data = ascii.read(filename)

        flat = util.resample_spectrum(self._wl_grid, data['wave'], data['flux'], fast=self.fast)

        self._project_2d_calibs(flat, 'flat', self.observation.flat_exptimes, self.observation.nflats,
                                overwrite=overwrite, list_lamps='00100')

    def create_arc(self, hg=False, ne=False, ar=False, xe=False, overwrite=True):

        if not np.any([hg, ne, ar, xe]):
            raise ValueError('At least one lamp should be selected')

        arcs = np.zeros((4, len(self._wl_grid)))

        list_lamps = ['0']*5

        if hg:
            filename = os.path.join(DATA_DIR, 'lamps', 'mercury_arc_sb.dat')
            data = ascii.read(filename)

            log.info('Including Hg lamp...')

            arcs[0] = util.resample_spectrum(self._wl_grid, data['wave'], data['flux'],
                                             fast=self.fast)
            list_lamps[3] = '1'

        if ne:
            filename = os.path.join(DATA_DIR, 'lamps', 'neon_arc_sb.dat')
            data = ascii.read(filename)

            log.info('Including Ne lamp...')

            arcs[1] = util.resample_spectrum(self._wl_grid, data['wave'], data['flux'], fast=self.fast)
            list_lamps[1] = '1'

        if ar:
            filename = os.path.join(DATA_DIR, 'lamps', 'argon_arc_sb.dat')
            data = ascii.read(filename)

            log.info('Including Ar lamp...')

            arcs[2] = util.resample_spectrum(self._wl_grid, data['wave'], data['flux'], fast=self.fast)
            list_lamps[0] = '1'

        if xe:
            filename = os.path.join(DATA_DIR, 'lamps', 'xenon_arc_sb.dat')
            data = ascii.read(filename)

            log.info('Including Xe lamp...')

            arcs[3] = util.resample_spectrum(self._wl_grid, data['wave'], data['flux'],
                                             fast=self.fast)
            list_lamps[4] = '1'

        # collapsing in the single required arc
        arcs = arcs.sum(axis=0)

        self._project_2d_calibs(arcs, 'arc', self.observation.arc_exptimes, self.observation.narcs, overwrite=overwrite,
                                list_lamps="".join(list_lamps))

    def create_bias(self, overwrite=True):

        self._project_2d_calibs(None, 'bias', 0, self.observation.nbias, overwrite=overwrite)

    def extract_extinction(self, extinction_file=os.path.join(DATA_DIR, 'sky',
                                                              'LVM_LVM160_KLAM.dat')):

        log.info('Reading the atmospheric extinction from file.')
        self.extinction_file = extinction_file
        data = ascii.read(self.extinction_file)

        self.extinction = util.resample_spectrum(self._wl_grid, data['col1'], data['col2'], fast=self.fast)

    def extract_sky(self, unit=u.erg / (u.cm ** 2 * u.s * u.AA)):
        """
        Return sky emission spectrum sampled at instrumental wavelengths
        """
        # science sky template
        flux, wave = util.open_sky_file(self.observation.sky_template,
                                        self.observation.days_moon, self.telescope.name,
                                        ha=self.observation.geocoronal,
                                        area=self._area_fiber.value)

        log.info('Resample sky emission to instrument wavelength solution.')
        self.sky = util.resample_spectrum(self._wl_grid, wave, flux, fast=self.fast) * unit

        if self.observation.sky1_template is not None:
            log.info('Using different template for sky array n. 1')

            flux1, wave1 = util.open_sky_file(self.observation.sky1_template,
                                            self.observation.days_moon, self.telescope.name,
                                            ha=self.observation.geocoronal,
                                            area=self._area_fiber.value)

            log.info('Resample sky emission to instrument wavelength solution.')
            self.sky1 = util.resample_spectrum(self._wl_grid, wave1, flux1, fast=self.fast) * unit

        if self.observation.sky2_template is not None:

            log.info('Using different template for sky array n. 1')
            flux2, wave2 = util.open_sky_file(self.observation.sky2_template,
                                            self.observation.days_moon, self.telescope.name,
                                            ha=self.observation.geocoronal,
                                            area=self._area_fiber.value)

            log.info('Resample sky emission to instrument wavelength solution.')
            self.sky2 = util.resample_spectrum(self._wl_grid, wave2, flux2, fast=self.fast) * unit

    def extract_target_spectra(self, unit=u.erg / (u.cm ** 2 * u.s * u.AA)):

        log.info(f"Recovering target spectra for {self.bundle.nfibers} fibers.")
        index, spectra = self.source.extract_spectra(self.bundle.fibers_science, self._wl_grid,
                                                     obs_coords=self.observation.target_coords)
        self.index = index
        self.target_spectra = spectra

    def extract_std_spectra(self, nstd=24, tmin=7000, tmax=8000, dt=250,
                            gmin=5, gmax=9, dg=0.1):

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
            stars.add_star(ra=0, dec=0, gmag=g, teff=T, ag=0, v=0, standard=True)

        stars.associate_spectra(shift=False, new_wave=self._wl_grid)
        stars.rescale_spectra()

        if self.save_std:
            outname = os.path.join(self.source.name, 'std_stars.fits.gz')
            stars.save_to_fits(outname=outname)

        self.standards = stars

    def simulate_science(self):

        unit_e = u.electron / (u.s * u.pix * u.cm**2)
        # get the target spectra and expand the array to cover all the unused fibers
        # I assume that the fiber rearranging will be performed at a later stage
        self.extract_target_spectra()
        spectra = np.zeros_like(self.target_spectra.value) * unit_e
        for i, spectrum in enumerate(self.target_spectra):
            spectra[i] = util.flam2epp(self._wl_grid, spectrum, self._disp) * u.electron

        # simulating spectra
        spectra *= self.telescope.aperture_area
        self.extract_extinction()
        extinction = expand_to_full_fiber(self.extinction, len(spectra))
        atmosphere = self.observation.airmass * extinction
        spectra *= 10 ** (-0.4 * (atmosphere))

        # simulating sky
        self.extract_sky()
        sky_e = util.flam2epp(self._wl_grid, self.sky, self._disp) * u.electron
        sky_fiber = sky_e * self.telescope.aperture_area

        sky = expand_to_full_fiber(sky_fiber, len(self.index))


        assert sky.shape == spectra.shape, 'Something wrong with sky and spectra'

        self.total_spectra = sky + spectra   # this is in electron
        # science_projection = np.zeros((self.bundle.max_obj_fibers, len(self._wl_grid)))
        # for i, spec in enumerate(self.total_spectra):
        #     science_projection[i] = spec

        # reshape sky spectra
        if self.sky1 is not None:
            sky1_e = util.flam2epp(self._wl_grid, self.sky1, self._disp) * u.electron
            sky1_fiber = sky1_e * self.telescope.aperture_area
            sky1 = expand_to_full_fiber(sky1_fiber, self.bundle.nfibers_sky1)
        else:
            sky1 = expand_to_full_fiber(sky_fiber, self.bundle.nfibers_sky1)

        if self.sky2 is not None:
            sky2_e = util.flam2epp(self._wl_grid, self.sky2, self._disp) * u.electron
            sky2_fiber = sky2_e * self.telescope.aperture_area
            sky2 = expand_to_full_fiber(sky2_fiber, self.bundle.nfibers_sky2)
        else:
            sky2 = expand_to_full_fiber(sky_fiber, self.bundle.nfibers_sky2)

        sky_projection = np.concatenate([sky1, sky2])

        # get standard stars spectra
        self.extract_std_spectra(nstd=self.bundle.nfibers_std)
        standards = np.zeros((self.bundle.nfibers_std, len(self._wl_grid))) * unit_e
        for i, std in enumerate(self.standards.spectra):
            standards[i] = util.flam2epp(self._wl_grid,
                                         std*u.erg / (u.cm ** 2 * u.s * u.AA),
                                         self._disp) * u.electron
        standards *= self.telescope.aperture_area
        extinction = expand_to_full_fiber(self.extinction, self.bundle.nfibers_std)
        atmosphere = self.observation.airmass * extinction
        standards *= 10 ** (-0.4 * atmosphere)

        sky_std = expand_to_full_fiber(sky_fiber, self.bundle.nfibers_std)

        standards += sky_std
        self._project_2d(self.total_spectra, sky_projection, standards)

    def _project_2d(self, science, sky, std):

        # get the tables for the different components of the final output
        sci_fibers = self.bundle.fibers_table_science

        ringid, pos, fibtype, fibid = get_fibers_table(science=sci_fibers)

        new_sky = {}
        new_sci = {}
        new_std = {}
        new_wave = {}
        # separating the full spectrum in the different branches

        for branch in self.spectrograph.branches:
            tmp_sky = sky * branch.efficiency(self._wl_grid)
            tmp_science = science * branch.efficiency(self._wl_grid)
            tmp_std = std * branch.efficiency(self._wl_grid)

            new_sky[branch.name], _ = reduce_size(tmp_sky, self._wl_grid,
                                                  branch.wavecoord.start, branch.wavecoord.end, delta_w=100*u.Angstrom)
            new_sci[branch.name], _ = reduce_size(tmp_science, self._wl_grid,
                                                  branch.wavecoord.start,branch.wavecoord.end, delta_w=100*u.Angstrom)
            new_std[branch.name], wave = reduce_size(tmp_std, self._wl_grid,
                                                     branch.wavecoord.start, branch.wavecoord.end,
                                                     delta_w=100*u.Angstrom)
            new_wave[branch.name] = wave

        for i, time in enumerate(self.observation.exptimes):
            for j, time_std in enumerate(self.observation.std_exptimes):
                log.info(f'Saving science exposures with {time}s exposures and {time_std}s '
                         'of exposure for each standard star')

                expname = (i*j)+j+1

                for branch in self.spectrograph.branches:
                    name = branch.name

                    sci_corr = new_sci[name] * time
                    sky_corr = new_sky[name] * time
                    std_corr = new_std[name] * time_std

                    spectra_final = np.vstack([sci_corr, sky_corr, std_corr])

                    self._to_camera(spectra=spectra_final, wave=new_wave[name],
                                    fibtype=fibtype, ringid=ringid, pos=pos, exptime=time,
                                    camera=name, exp_type='science', branch=branch,
                                    exp_name=expname)

    def _project_2d_calibs(self, data, calib_name, exptimes, nexpo, overwrite=True, list_lamps='00000'):

        if not isinstance(exptimes, list):
            exptimes = [exptimes]

        # moving from SB to flux
        # the arc files are already in e/s/pix/arcsec^2 no need to tranform and multiply
        # per telescope area
        new_calib = {}
        new_wave = {}
        if calib_name != 'bias':
            calib = data * self._area_fiber.value

            # for now the calibrations are independent of the position of the fibers in the field
            # THIS WILL CHANGE IN THE FUTURE.

            ringid, pos, fibtype, fibid = get_fibers_table(science=None)

            # applying the sensitivity function and reducing the size to spare memory
            for branch in self.spectrograph.branches:
                tmp = calib * branch.efficiency(self._wl_grid)
                resized, wave = reduce_size(tmp, self._wl_grid,
                                            branch.wavecoord.start, branch.wavecoord.end, delta_w=100 * u.Angstrom)
                new_calib[branch.name] = resized
                new_wave[branch.name] = wave

        if calib_name == 'arc':
            add_file_index = 10001
        elif calib_name == 'flat':
            add_file_index = 1001
        elif calib_name == 'bias':
            add_file_index = 101
        else:
            log.error(f"Unrecognized calibration name is detected: {calib_name}")
            return

        channel_index = {'blue': 'b', 'red': 'r', 'ir': 'z'}
        for i, time in enumerate(exptimes):
            for j in range(nexpo):
                exp_name = add_file_index+(i*j)+j

                log.info(f'Saving {calib_name} exposures with exptime {time}s')
                for branch in self.spectrograph.branches:
                    name = branch.name
                    if calib_name == 'bias':
                        calib_final = None
                        new_wave[name] = None
                        fibtype = None
                        ringid = None
                        pos = None
                    else:
                        calib = new_calib[name] * time
                        calib_final = expand_to_full_fiber(calib, self.bundle.max_fibers)
                    # calib_final = calib_final.T

                    fileout_root = f'sdR-s-{channel_index[name]}1-'+f'{exp_name:08}'[:-len(str(add_file_index))+1]
                    if not overwrite and (i == 0) and (j == 0):
                        files_exist = glob.glob(os.path.join(self.outdir,
                                                             f'{fileout_root}*.fits.gz'))
                        if len(files_exist) > 0:
                            exp_name += max([(int(curfile.split('-')[-1].split('.fits.gz')[0]) -
                                              add_file_index + 1) for curfile in files_exist])

                    self._to_camera(spectra=calib_final, wave=new_wave[name],
                                    fibtype=fibtype, ringid=ringid, pos=pos, exptime=time,
                                    camera=name, exp_type=calib_name, branch=branch,
                                    exp_name=exp_name, list_lamps=list_lamps)

    def _to_camera(self, spectra, wave, fibtype, ringid, pos, exptime, camera, exp_type,
                   branch, exp_name, list_lamps='00000'):

        n_cr = int(branch.cosmic_rates.value * exptime)

        if camera == 'blue':
            expn = '00002998'
            cam = 'b1'
        elif camera == 'red':
            expn = '00001563'
            cam = 'r1'
        elif camera == 'ir':
            expn = '00001563'
            cam = 'z1'
        else:
            log.error(f"Unrecognized spectrograph branch: {camera}")
            return
        cube_file = f'{DATA_DIR}/instrument/sdR-s-{cam}-{expn}.disp.fits'
        wave2d, _ = fits.getdata(cube_file, 0, header=True)
        wave_ccd = wave2d[3]

        channel_index = {'blue': 'b', 'red': 'r', 'ir': 'z'}
        for cam in range(3):
            projected_spectra = cre_raw_exp(spectra, fibtype=fibtype, ring=ringid,
                                            position=pos, wave_ccd=wave_ccd, wave=wave,
                                            nfib=self._fibers_per_spec, channel_type=camera,
                                            cam=cam+1, n_cr=n_cr, exp_name=exp_name,
                                            exp_time=exptime,
                                            ra=self.observation.ra, dec=self.observation.dec,
                                            obstime=str(self.observation.time),
                                            mjd=str(self.observation.mjd),
                                            flb=exp_type, add_cr_hits=n_cr > 0,
                                            list_lamps=list_lamps)
            if projected_spectra is not None:
                fileout_data = os.path.join(self.outdir, f'sdR-s-{channel_index[camera]}{cam+1}-{exp_name:08}.fits')
                if self.compress:
                    fileout_data += '.gz'
                projected_spectra[0].writeto(fileout_data, overwrite=True)
                log.info(f"Done for camera #{cam+1} and {camera} branch")
            else:
                log.error(f"Something went wrong for #{cam + 1} and {camera} branch")

