import numpy as np
import astropy.units as u

from lvmdatasimulator.field import LVMField
from lvmdatasimulator.observation import Observation
from lvmdatasimulator.telescope import LVM160
from lvmdatasimulator.instrument import LinearSpectrograph
from lvmdatasimulator.simulator import Simulator
from lvmdatasimulator.fibers import FiberBundle
from lvmdatasimulator import log, WORK_DIR
from astropy.io.misc import yaml
from astropy.io import ascii
from astropy.io import fits
from matplotlib import pyplot as plt

import shutil
import time
import os


def save_input_params(params):
    """
    Print the values used to run the simulator to file

    Args:
        params (dict):
        dictionary containing all the parameters used to run the simulator.
    """

    default = dict(# LVMField inputs
                   ra=10,
                   dec=-10,
                   size=32,
                   pxsize=1,
                   unit_ra=u.degree,
                   unit_dec=u.degree,
                   unit_size=u.arcmin,
                   unit_pxsize=u.arcsec,
                   name='LVM_Field',

                   # Nebulae generation
                   nebulae=None,
                   nebulae_name="LVM_field_nebulae",
                   nebulae_from_file=None,

                   # Star list generation
                   gmag_limit=17,
                   shift=False,
                   save=True,
                   starlist_name=None,

                   # save input map
                   wavelength_ranges=[6550, 6570],
                   unit_range=u.AA,

                   # parameters of observations
                   ra_bundle=10,
                   dec_bundle=-10,
                   unit_ra_bundle=u.deg,
                   unit_dec_bundle=u.deg,
                   time='2022-01-01T00:00:00.00',  # UT time of observations
                   utcoffset=-4 * u.hour,  # Correction to local time. It is important to keep it updated
                   exptimes=[900],  # exposure times in s
                   airmass= None,
                   days_moon= None,
                   sky_template= None,

                   # bundle properties
                   bundle_name='full',
                   nrings=24,
                   angle=0,
                   custom_fibers=None,

                   # parameters of the simulator
                   fast=True
                   )

    # filling missing keys with default values
    for key in default.keys():
        if key not in params:
            if key in ['ra_bundle', 'dec_bundle']:
                params[key] = params.get(key.replace('_bundle', ''), default[key])
            else:
                params[key] = default[key]

    outname = '{}_input_parameters.yml' .format(params['name'])
    log.info(f'Saving input parameters to {outname}')

    outname = os.path.join(WORK_DIR, params['name'], outname)  # putting it in the WORK/NAME directory
    with open(outname, 'w') as fp:
        yaml.dump(params, fp)


def save_input_params_etc(params):

    default = {'name': 'LVM_Field_ETC',
               'spectrum': None,
               'norm': 1,
               'lsf_fwhm': 1.5,
               'airmass': 1.5,
               'unit_wave': u.AA,
               'unit_flux': u.erg*u.s**-1*u.cm**-2*u.arcsec**-2,
               'days_moon': 0,
               'sky_template': None}


    for key in default.keys():
        if key not in params:
            params[key] = default[key]

    outname = '{}_input_parameters_etc.yml' .format(params['name'])
    log.info(f'Saving input ETC parameters to {outname}')

    outname = os.path.join(WORK_DIR, params['name'], outname)  # putting it in the WORK/NAME directory
    with open(outname, 'w') as fp:
        yaml.dump(params, fp)


def open_input_params(filename):

    log.info(f'Reading input parameters from {filename}')
    filename = os.path.join(WORK_DIR, filename)
    with open(filename, 'r') as fp:
        params = yaml.load(fp)

    return params


def run_test(name='LVMsimulator_test'):
    """
    Run simple simulation as the test
    """
    if name == 'LVMsimulator_test':
        name = name + f"_{os.environ['USER']}"
    my_nebulae = [{"type": 'DIG', 'max_brightness': 1e-17,
                   'perturb_amplitude': 0.1, 'perturb_scale': 200 * u.pc},
                  {'type': 'Bubble', 'max_brightness': 8e-16, 'thickness': 0.2, 'radius': 40,
                   'expansion_velocity': 30, 'sys_velocity': 20,
                   'n_brightest_lines': 20,
                   'model_params': {'Z': 1., 'Teff': 65000, 'LogLsun': 5., 'nH': 150, 'Geometry': 'Shell'},
                   'model_type': 'cloudy', 'offset_RA': 0, 'offset_DEC': 0}
                  ]
    parameters = dict(
        # LVMField inputs
        ra=10,  # RA of the source field
        dec=-10,  # DEC of the source field
        size=15,  # size of the source field
        pxsize=1,  # spaxel size of the source field
        unit_ra=u.degree,  # unit of RA
        unit_dec=u.degree,  # unit of DEC
        unit_size=u.arcmin,  # unit of size
        unit_pxsize=u.arcsec,  # unit of spaxel
        name=name,  # name of the field

        # Nebulae generation
        nebulae=my_nebulae,  # list of dictionaries defining the nebulae in the field
        nebulae_name=f"{name}_nebulae.fits",  # name of the output source field file

        # Star list generation
        gmag_limit=18,  # maximum magnitude of gaia stars to include in the field

        # save input/output map
        wavelength_ranges=[[6550, 6570]],  # wavelength ranges to integrate for the input and output maps
        unit_range=u.AA,  # units of measurement of the wavelength ranges

        # parameters of Observation
        ra_bundle=10,  # RA of the center of the bundle
        dec_bundle=-10,  # DEC of the center of the bundle
        unit_ra_bundle=u.deg,  # unit of ra_bundle
        unit_dec_bundle=u.deg,  # unit of dec_bundle
        exptimes=[24 * 900],  # list of exposure times in s

        # bundle properties
        bundle_name='full',  # type of fiber configuration
        nrings=8,  # number of rings to simulate
        angle=0,  # rotation to apply to the bundle.

        # parameters of the simulator
        fast=True  # use normal interpolation or precise resampling.
    )
    log.info('Start test simulations. It should take several minutes')
    run_simulator_1d(parameters)


def run_simulator_1d(params):
    """
    Main function to run the simulation of an LVM field.

    Args:
        params (dict, str):
        Dictionary containing all the input needed to run the simulator or name of the JSON file
        where it is stored.
    """

    if isinstance(params, str):
        params = open_input_params(params)

    start = time.time()
    my_lvmfield = LVMField(ra=params.get('ra', 10),
                           dec=params.get('dec', -10),
                           size=params.get('size', 32),
                           pxsize=params.get('pxsize', 1),
                           unit_ra=params.get('unit_ra', u.degree),
                           unit_dec=params.get('unit_dec', u.degree),
                           unit_size=params.get('unit_size', u.arcmin),
                           unit_pxsize=params.get('unit_pxsize', u.arcsec),
                           name=params.get('name', 'LVM_Field'))

    if params.get('nebulae_from_file', None) is not None:
        my_lvmfield.add_nebulae(load_from_file=params['nebulae_from_file'])
    else:
        if params['nebulae'] is None:
            raise ValueError('No nebulae defined, aborting the simulation')

        my_lvmfield.add_nebulae(params['nebulae'],
                                save_nebulae=params.get('nebulae_name', 'LVM_field_nebulae'))

    my_lvmfield.generate_gaia_stars(gmag_limit=params.get('gmag_limit', 17),
                                    shift=params.get('shift', False),
                                    save=params.get('save', True),
                                    filename=params.get('starlist_name', None))

    my_lvmfield.get_map(wavelength_ranges=params.get('wavelength_ranges', [[6550, 6570]]),
                        unit_range=params.get('unit_range', u.AA))

    obs = Observation(name=params.get('name', 'LVM_field'),
                      ra=params.get('ra_bundle', params.get('ra', 10)),
                      dec=params.get('dec_bundle', params.get('dec', -10)),
                      unit_ra=params.get('unit_ra_bundle', u.deg),
                      unit_dec=params.get('unit_dec_bundle', u.deg),
                      time=params.get('time', '2022-01-01T00:00:00.00'),
                      utcoffset=params.get('utcoffset', -3 * u.hour),
                      exptimes=params.get('exptimes', 900.0),
                      airmass=params.get('airmass', None),
                      days_moon=params.get('days_moon', None),
                      sky_template=params.get('sky_template', None))
    tel = LVM160()
    spec = LinearSpectrograph()
    bundle = FiberBundle(bundle_name=params.get('bundle_name', 'full'),
                         nrings=params.get('nrings', 24),
                         angle=params.get('angle', 0),
                         custom_fibers=params.get('custom_fibers', None))

    sim = Simulator(my_lvmfield, obs, spec, bundle, tel, fast=params.get('fast', True))
    sim.simulate_observations()
    sim.save_outputs()
    sim.save_output_maps(wavelength_ranges=params.get('wavelength_ranges', [6550, 6570]),
                         unit_range=params.get('unit_range', u.AA))

    save_input_params(params)

    log.info('Done. Elapsed time: {:0.1f}'.format(time.time()-start))


def run_lvm_etc(params, check_lines=None, desired_snr=None, continuum=False, delete=True, dlam=1):
    """
        Simple run the simulations in the mode of exposure time calculator.

        Args:
            params (dict, str):
                Dictionary containing all the input needed to run the simulator or name of the JSON file
                where it is stored.
            check_lines (float, list, tuple):
                Wavelength or the list of wavelength to examine in the output. Default - only Halpha line
            desired_snr (float, list, tuple):
                Desired signal-to-noise ratios in corresponding lines. Should be of the same size as check_lines
            continuum (bool):
                If True, then S/N will be measured assuming the source spectrum is continuum (signal = flux per pixel);
                otherwise (default) the flux is integrated in the window of ±2A assuming the
                emission line spectrum(noise is also scaled)
            delete (bool):
                delete all the output files. Defaults to True.
    """

    if isinstance(params, str):
        params = open_input_params(params)
    if ('name' not in params) or not isinstance(params.get('name'), str):
        params['name'] = 'LVM_ETC'
    if ('nebula' not in params) or (type(params['nebula']) is not dict):
        nebula = None
    else:
        nebula = params['nebula']
        if nebula.get('max_brightness') is None or nebula.get('max_brightness') < 0:
            nebula = None

    if ('star' not in params) or (type(params['star']) is not dict):
        star = None
    else:
        star = params['star']

    spectrum_name = params.get('spectrum', None)

    if not isinstance(spectrum_name, str) and spectrum_name is not None:
        raise TypeError(f'"spectrum" can be string or None. It is {type(params["spectrum"])}')

    if (spectrum_name is not None) and (nebula is not None or star is not None):
        raise ValueError(f'"spectrum cannot be used with other sources')

    if star is None and nebula is None and spectrum_name is None:
        raise ValueError('Neither nebula, nor star nor spectrum are defined, '
                         'or they are defined incorrectly. Aborting the simulation')

    str_print = 'Start simulations in exposure time calculator mode for '
    if nebula is not None:
        str_print += '1 nebula '
        if star is not None:
            str_print += 'and '
    if star is not None:
        str_print += '1 star '
    if spectrum_name is not None:
        str_print += '1 custom spectrum.'

    log.info(str_print)

    if check_lines is None:
        check_lines = [6563.]
    if type(check_lines) in [float, int]:
        check_lines = [check_lines]

    start = time.time()
    name = params.get('name', 'LVM_Field_ETC')
    my_lvmfield = LVMField(ra=10,
                           dec=-10,
                           size=1,
                           pxsize=1,
                           unit_ra=u.deg,
                           unit_dec=u.deg,
                           unit_size=u.arcmin,
                           unit_pxsize=u.arcsec,
                           name=name)

    if nebula is not None:
        inp_nebulae = [{"type": 'DIG',
                        'perturb_scale': 0, 'perturb_amplitude': 0,
                        'max_brightness': nebula.get('max_brightness'),
                        'model_id': nebula.get('model_id'),
                        'model_params': nebula.get('model_params'),
                        'model_type': nebula.get('model_type', 'cloudy'),
                        'continuum_type': nebula.get('continuum_type'),
                        'continuum_data': nebula.get('continuum_data'),
                        'continuum_mag': nebula.get('continuum_mag'),
                        'continuum_flux': nebula.get('continuum_flux', 0),
                        'continuum_wl': nebula.get('continuum_wl', 5500.),
                        'offset_X': 0, 'offset_Y': 0}]

        if params.get('Av', None) is not None or params.get('ebv', None) is not None:
            if params.get('ebv', None) is not None:
                av = params.get('ebv', None) * 3.1 * u.mag
            else:
                av = params.get('Av', None) * u.mag

            log.info(f'Adding extinction: A(V) = {av:0.2f}')

            inp_nebulae.append({'type': 'Circle',
                                'distance': 1 * u.kpc,
                                'radius': 0.4 * u.pc,
                                'max_extinction': av,
                                'ext_law': 'CCM89',
                                'zorder': 4, 'offset_X': 0, 'offset_Y': 0})
        my_lvmfield.add_nebulae(inp_nebulae, save_nebulae='test.fits')

    if star is not None:
        if not continuum:
            log.warning('A star is included in the simulation. The S/N estimate might not be reliable \n'
                        'or the ETC could crash if the considered lines sit on top of an absorption lines.')
        my_lvmfield.generate_single_stars(parameters=star)

    default_exptimes = list(np.round(np.logspace(np.log10(10), np.log10(90000), 15)).astype(int))
    exptimes = params.get('exptimes', default_exptimes)
    obs = Observation(name=name,
                      ra=10,
                      dec=-10,
                      unit_ra=u.deg,
                      unit_dec=u.deg,
                      exptimes=exptimes,
                      airmass=params.get('airmass', 1.5),
                      days_moon=params.get('days_moon', 0),
                      sky_template=params.get('sky_template', None),
                      geocoronal=params.get('geocoronal', None))

    tel = LVM160()
    spec = LinearSpectrograph()
    bundle = FiberBundle(bundle_name='central')
    sim = Simulator(my_lvmfield, obs, spec, bundle, tel, fast=True, aperture=10*u.pix)

    if spectrum_name is not None:
        data = ascii.read(spectrum_name)
        wave = data['col1']
        flux = data['col2']
        unit_flux = u.erg*u.s**-1*u.cm**-2*u.arcsec**-2*u.AA**-1
        sim.simulate_observations_custom_spectrum(wave, flux, norm=params.get('norm', 1),
                                                  unit_wave=params.get('unit_wave', u.AA),
                                                  unit_flux=params.get('unit_flux', unit_flux))
    else:
        sim.simulate_observations()

    sim.save_outputs()

    save_input_params_etc(params)

    outdir = os.path.join(WORK_DIR, params['name'], 'outputs')

    snr_output = np.zeros(shape=(len(check_lines), len(exptimes)))
    for exp_id, exptime in enumerate(exptimes):
        outname = os.path.join(outdir, f'{name}_linear_central_{exptime}_no_noise.fits')
        with fits.open(outname) as hdu:
            for l_id, line in enumerate(check_lines):
                rec_wl = (hdu['WAVE'].data > (line - dlam)) & (hdu['WAVE'].data < (line + dlam))
                if continuum:
                    snr_output[l_id, exp_id] = np.nanmean(hdu['SNR'].data[0,
                                                            (hdu['WAVE'].data > (line - dlam)) &
                                                            (hdu['WAVE'].data < (line + dlam))])
                else:
                    rec_wl_cnt = (hdu['WAVE'].data > (line - dlam*30)) & (hdu['WAVE'].data < (line + dlam*30))
                    flux = np.nansum(hdu['TARGET'].data[0, rec_wl] - np.nanmedian(hdu['TARGET'].data[0, rec_wl_cnt]))
                    if flux < 0:
                        flux = 0

                    snr_output[l_id, exp_id] = flux/np.sqrt(np.nansum(hdu['ERR'].data[0, rec_wl]**2))

    if desired_snr is not None and (len(desired_snr) == len(check_lines)):
        desired_exptimes = []
    else:
        desired_exptimes = None

    if params.get('exptimes', None) is None:
        fig, ax = plt.subplots()
        for l_id, line in enumerate(check_lines):

            ax.scatter(exptimes, snr_output[l_id, :], label=str(line))

            res = np.polyfit(np.log10(snr_output[l_id, :]), np.log10(exptimes), 3)
            p = np.poly1d(res)
            ax.plot(10**p(np.log10(snr_output[l_id, :])), snr_output[l_id, :])

            if desired_snr is not None and (len(desired_snr) == len(check_lines)):

                desired_exptimes.append(np.round(10**p(np.log10(desired_snr[l_id]))).astype(int))
                print(f'To reach S/N={desired_snr[l_id]} in line = {line}±{dlam}A we need '
                    f'{desired_exptimes[-1]}s of single exposure')
                ax.scatter(desired_exptimes[-1], desired_snr[l_id], c='r', marker='*',
                           s=200, zorder=100)

        ax.set_xscale('log')
        ax.set_yscale('log')

        ax.legend(loc='best')

        ax.set_xlabel("Exposure time, s")
        ax.set_ylabel("Expected S/N ratio")

        plt.show()
    else:
        for exp_id, exptime in enumerate(exptimes):
            for l_id, line in enumerate(check_lines):
                print(f'The S/N reached in a {exptime}s exposure in line = {line}±{dlam}A is '+\
                      f'{snr_output[l_id, exp_id]:0.2f}')

            outname = os.path.join(outdir, f'{name}_linear_central_{exptime}_no_noise.fits')
            with fits.open(outname) as hdu:
                wave = hdu['WAVE'].data
                flux = hdu['TARGET'].data[0]
                err = hdu['ERR'].data[0]
                snr = hdu['SNR'].data[0]

            fig, ax = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
            ax[0].plot(wave, np.log10(flux), c='k', label='Flux')
            ax[0].plot(wave, np.log10(err), c='r', label='Error')
            ax[1].plot(wave, np.log10(snr), c='b')
            ax[1].axhline(np.log10(3), ls='--', c='orange', label='S/N=3')

            ax[0].legend(loc='best')
            ax[1].legend(loc='best')

            ax[0].set_ylabel('Flux (e/pix)')
            ax[1].set_ylabel('S/N')
            ax[1].set_xlabel('Wavelength ($\\AA$)')
            plt.subplots_adjust(hspace=0)
            plt.show()


    if delete:
        log.info('Deleting output directory')
        # not sure about what is the best solution
        # shutil.rmtree(os.path.join(WORK_DIR, params['name']), ignore_errors=True)
        shutil.rmtree(outdir, ignore_errors=True)

    print('\nElapsed time: {:0.1f}s' .format(time.time() - start))
    return desired_exptimes
