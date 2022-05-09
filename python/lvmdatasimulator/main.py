from email.policy import default
from lvmdatasimulator.field import LVMField
from lvmdatasimulator.observation import Observation
from lvmdatasimulator.telescope import LVM160
from lvmdatasimulator.instrument import LinearSpectrograph
from lvmdatasimulator.simulator import Simulator
from lvmdatasimulator.fibers import FiberBundle
from astropy.table import Table
# import imageio

import astropy.units as u
import time
import argparse


def run_simulator(params):

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
        my_lvmfield.add_nebulae(params['nebulae'],
                                save_nebulae=params.get('nebulae_name', 'LVM_field_nebulae'))

    my_lvmfield.generate_gaia_stars(gmag_limit=params.get('gmag_limit', 17),
                                    shift=params.get('shift', False),
                                    save=params.get('save', True),
                                    filename=params.get('starlist_name', None))

    my_lvmfield.get_map(wavelength_range=params.get('wavelength_range', [6550, 6570]),
                        unit_range=params.get('unit_range', u.AA),
                        save_file=params.get('save_file', 'outputs/LVM_Field_input.fits'))

    obs = Observation(name=params.get('name', 'LVM_field'),
                      ra=params.get('ra_bundle', 10),
                      dec=params.get('dec_bundle', -10),
                      unit_ra=params.get('unit_ra_bundle', u.deg),
                      unit_dec=params.get('unit_dec_bundle', u.deg),
                      time=params.get('time', '2022-01-01T00:00:00.00'),
                      utcoffset=params.get('utcoffset', -3 * u.hour),
                      exptime=params.get('exptime', 900.0 * u.s),
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
    sim.save_output_maps(wavelength_range=params.get('wavelength_range', [6550, 6570]),
                         unit_range=params.get('unit_range', u.AA))
    print('Elapsed time: {:0.1f}' .format(time.time()-start))


