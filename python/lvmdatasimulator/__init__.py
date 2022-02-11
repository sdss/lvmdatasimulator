# encoding: utf-8
import os
from sdsstools import get_config, get_logger, get_package_version
import warnings

# pip package name
NAME = 'sdss-lvmdatasimulator'

# Loads config. config name is the package name.
config = get_config('lvmdatasimulator', config_envvar="LVMDSIM_CONFIG")

# Inits the logging system as NAME. Only shell logging, and exception and warning catching.
# File logging can be started by calling log.start_file_logger(path).  Filename can be different
# than NAME.
log = get_logger(NAME)

# import root directory
ROOT_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), '..', '..'))

# maximal number of processes for parallelization
n_process = config.get('nprocess')

# path to the Cloudy models and some default parameters
if config.get('data_dir').startswith("/") or config.get('data_dir').startswith("\\"):
    CLOUDY_MODELS = os.path.join(config.get('data_dir'), config.get('cloudy_models_name'))
else:
    CLOUDY_MODELS = os.path.join(os.path.join(ROOT_DIR, config.get('data_dir')), config.get('cloudy_models_name'))
if not os.path.isfile(CLOUDY_MODELS):
    log.warning("Pre-computed grid of Cloudy models ({}) is not found. "
                "Flux distribution in different lines will be unavailable".format(CLOUDY_MODELS))
    CLOUDY_MODELS = None
CLOUDY_SPEC_DEFAULTS = config.get('cloudy_default_params')

# path to the directory where all computational results will be saved
if config.get('run_dir').startswith("/") or config.get('run_dir').startswith("\\"):
    WORK_DIR = config.get('run_dir')
else:
    WORK_DIR = os.path.join(ROOT_DIR, config.get('run_dir'))

warnings.filterwarnings('ignore', r'divide by zero encountered in')
warnings.filterwarnings('ignore', r'invalid value encountered in ')

# package name should be pip package name
__version__ = get_package_version(path=__file__, package_name=NAME)


