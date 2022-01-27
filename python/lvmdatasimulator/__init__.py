# encoding: utf-8
import os
from sdsstools import get_config, get_logger, get_package_version
import warnings

# pip package name
NAME = 'sdss-lvmdatasimulator'

# Loads config. config name is the package name.
config = get_config('lvmdatasimulator')

# Inits the logging system as NAME. Only shell logging, and exception and warning catching.
# File logging can be started by calling log.start_file_logger(path).  Filename can be different
# than NAME.
log = get_logger(NAME)

# import root directory
ROOT_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), '..', '..'))

# maximal number of processes for parallelization
n_process = 8

# path to the Cloudy models and some default parameters
CLOUDY_MODELS = os.path.join(os.path.join(ROOT_DIR, 'data'), 'LVM_cloudy_sphere_models.fits')
if not os.path.isfile(CLOUDY_MODELS):
    log.warning("Pre-computed grid of Cloudy models ({}) is not found. "
                "Flux distribution in different lines will be unavailable".format(CLOUDY_MODELS))
    CLOUDY_MODELS = None
CLOUDY_SPEC_DEFAULTS = {'id': 901, 'Z': 0.3, 'qH': 49., 'Teff': 70000., 'nH': 30., 'Rin': 10.01}

warnings.filterwarnings('ignore', r'divide by zero encountered in')
warnings.filterwarnings('ignore', r'invalid value encountered in ')

# package name should be pip package name
__version__ = get_package_version(path=__file__, package_name=NAME)


