from lvmdatasimulator.field import LVMField
from lvmdatasimulator.observation import Observation
from lvmdatasimulator.telescope import LVM160
from lvmdatasimulator.instrument import LinearSpectrograph
from lvmdatasimulator.simulator import Simulator
from lvmdatasimulator.fibers import FiberBundle
from lvmdatasimulator import log, WORK_DIR
from astropy.io.misc import yaml

import astropy.units as u
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
                   spaxel=1,
                   unit_ra=u.degree,
                   unit_dec=u.degree,
                   unit_size=u.arcmin,
                   unit_spaxel=u.arcsec,
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
    for key in params.keys():
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
                           spaxel=params.get('spaxel', 1),
                           unit_ra=params.get('unit_ra', u.degree),
                           unit_dec=params.get('unit_dec', u.degree),
                           unit_size=params.get('unit_size', u.arcmin),
                           unit_spaxel=params.get('unit_spaxel', u.arcsec),
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


