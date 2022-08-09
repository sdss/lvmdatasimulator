import numpy as np

from lvmdatasimulator.field import LVMField
from lvmdatasimulator.observation import Observation
from lvmdatasimulator.telescope import LVM160
from lvmdatasimulator.instrument import LinearSpectrograph
from lvmdatasimulator.simulator import Simulator
from lvmdatasimulator.fibers import FiberBundle
from lvmdatasimulator import log, WORK_DIR
from astropy.io.misc import yaml
from astropy.io import ascii

import astropy.units as u
import time
from astropy.io import fits
from matplotlib import pyplot as plt
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

    outname = os.path.join(WORK_DIR, outname)  # putting it in the WORK directory
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
               'days_moon': 0,}

    for key in default.keys():
        if key not in params:
            params[key] = default[key]

    outname = '{}_input_parameters_etc.yml' .format(params['name'])
    log.info(f'Saving input ETC parameters to {outname}')

    outname = os.path.join(WORK_DIR, outname)  # putting it in the WORK directory
    with open(outname, 'w') as fp:
        yaml.dump(params, fp)


def open_input_params(filename):

    log.info(f'Reading input parameters from {filename}')
    filename = os.path.join(WORK_DIR, filename)
    with open(filename, 'r') as fp:
        params = yaml.load(fp)

    return params


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

    print('Elapsed time: {:0.1f}' .format(time.time()-start))


def run_lvm_etc(params, check_lines=None, desired_snr=None, continuum=False, delete=True):
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
    """

    if isinstance(params, str):
        params = open_input_params(params)

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

    if not isinstance(spectrum_name, str) and not spectrum_name is None:
        raise TypeError(f'"spectrum" can be string or None. It is {type(params["spectrum"])}')

    if (spectrum_name is not None) and (nebula is not None or star is not None):
        raise ValueError(f'"spectrum cannot be used with other sources')

    if star is None and nebula is None and spectrum_name is None:
        raise ValueError('Neither nebula, nor star nor spectrum are defined, or they are defined incorrectly. Aborting the simulation')

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
        my_lvmfield.add_nebulae([{"type": 'DIG',
                                  'perturb_scale': 0, 'perturb_amplitude': 0,
                                  'max_brightness': nebula.get('max_brightness'),
                                  'cloudy_id': nebula.get('cloudy_id'),
                                  'cloudy_params': nebula.get('cloudy_params'),
                                  'continuum_type': nebula.get('continuum_type'),
                                  'continuum_data': nebula.get('continuum_data'),
                                  'continuum_mag': nebula.get('continuum_mag'),
                                  'continuum_flux': nebula.get('continuum_flux', 0),
                                  'continuum_wl': nebula.get('continuum_wl', 5500.),
                                  'offset_X': 0, 'offset_Y': 0}])
    if star is not None:
        my_lvmfield.generate_single_stars(parameters=star)

    default_exptimes = list(np.round(np.logspace(np.log10(300), np.log10(90000), 15)).astype(int))
    exptimes = params.get('exptimes', default_exptimes)
    obs = Observation(name=name,
                      ra=10,
                      dec=-10,
                      unit_ra=u.deg,
                      unit_dec=u.deg,
                      exptimes=exptimes,
                      airmass=params.get('airmass', 1.5),
                      days_moon=params.get('days_moon', 0))

    tel = LVM160()
    spec = LinearSpectrograph()
    bundle = FiberBundle(bundle_name='central')
    sim = Simulator(my_lvmfield, obs, spec, bundle, tel, fast=True)

    if spectrum_name is not None:
        data = ascii.read(spectrum_name)
        wave = data['col1']
        flux = data['col2']
        sim.simulate_observations_custom_spectrum(wave, flux,
                                                  lsf_fwhm=params.get('lsf_fwhm', 1.5),
                                                  norm=params.get('norm', 1))
    else:
        sim.simulate_observations()

    sim.save_outputs()

    save_input_params_etc(params)

    outdir = os.path.join(WORK_DIR, 'outputs')

    snr_output = np.zeros(shape=(len(check_lines), len(exptimes)))
    w_lam = 2.
    for exp_id, exptime in enumerate(exptimes):
        outname = os.path.join(outdir, f'{name}_linear_central_{exptime}_flux.fits')
        with fits.open(outname) as hdu:
            for l_id, line in enumerate(check_lines):
                if continuum:
                    snr_output[l_id, exp_id] = np.nanmax(hdu['SNR'].data[0,
                                                                         (hdu['WAVE'].data > (line - w_lam)) &
                                                                         (hdu['WAVE'].data < (line + w_lam))])
                else:
                    rec_wl = (hdu['WAVE'].data > (line - w_lam)) & (hdu['WAVE'].data < (line + w_lam))
                    rec_wl_cnt = (hdu['WAVE'].data > (line - w_lam*30)) & (hdu['WAVE'].data < (line + w_lam*30))
                    flux = np.nansum(hdu['TARGET'].data[0, rec_wl] - np.nanmedian(hdu['TARGET'].data[0, rec_wl_cnt]))
                    if flux < 0:
                        flux = 0
                    snr_output[l_id, exp_id] = flux/np.sqrt(np.nansum(hdu['ERR'].data[0, rec_wl]**2))
                    # snr_output[l_id, exp_id] = np.nanmax(hdu['SNR'].data[0,
                    #                                                      (hdu['WAVE'].data > (line - w_lam)) &
                    #                                                      (hdu['WAVE'].data < (line + w_lam))])
        if delete:
            os.remove(outname)  # remove temporary files
            os.remove(outname.replace('flux', 'realization'))
            os.remove(outname.replace('flux', 'no_noise'))

    if delete:
        os.remove(os.path.join(outdir, f'{name}_linear_central_input.fits'))
        # remove output directory if empty
        if len(os.listdir(outdir)) == 0:
            os.rmdir(outdir)

    if desired_snr is not None and (len(desired_snr) == len(check_lines)):
        desired_exptimes = []
    else:
        desired_exptimes = None

    fig, ax = plt.subplots()
    for l_id, line in enumerate(check_lines):

        ax.scatter(exptimes, snr_output[l_id, :], label=str(line))

        res = np.polyfit(np.log10(snr_output[l_id, :]), np.log10(exptimes), 3)
        p = np.poly1d(res)
        ax.plot(10**p(np.log10(snr_output[l_id, :])), snr_output[l_id, :])

        if desired_snr is not None and (len(desired_snr) == len(check_lines)):

            desired_exptimes.append(np.round(10**p(np.log10(desired_snr[l_id]))).astype(int))
            print(f'To reach S/N={desired_snr[l_id]} in line = {line}±{w_lam}A we need '
                  f'{desired_exptimes[-1]}s of single exposure')

    ax.set_xscale('log')
    ax.set_yscale('log')

    ax.legend(loc='best')

    ax.set_xlabel("Exposure time, s")
    ax.set_ylabel("Expected S/N ratio")

    plt.show()
    print('\nElapsed time: {:0.1f}s' .format(time.time() - start))
    return desired_exptimes
