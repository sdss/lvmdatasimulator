# encoding: utf-8
import os
from sdsstools import get_config, get_logger, get_package_version
import warnings
from urllib.request import urlretrieve

# pip package name
NAME = 'sdss-lvmdatasimulator'

# Loads config. config name is the package name.
config = get_config('lvmdatasimulator', config_envvar="LVM_SIMULATOR_CONFIG")

# Inits the logging system as NAME. Only shell logging, and exception and warning catching.
# File logging can be started by calling log.start_file_logger(path).  Filename can be different
# than NAME.
log = get_logger(NAME)

# import root directory
ROOT_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), '..', '..'))

# maximal number of processes for parallelization
n_process = config.get('nprocess')

if config.get('data_dir').startswith("."):
    DATA_DIR = os.path.curdir
elif config.get('data_dir').startswith("/") or config.get('data_dir').startswith("\\"):
    DATA_DIR = config.get('data_dir')
else:
    DATA_DIR = os.path.join(ROOT_DIR, config.get('data_dir'))

# path to the Cloudy models and some default parameters
if config.get('data_dir').startswith("."):
    CLOUDY_MODELS = os.path.join(os.path.curdir, config.get('cloudy_models_name'))
elif config.get('data_dir').startswith("/") or config.get('data_dir').startswith("\\"):
    CLOUDY_MODELS = os.path.join(config.get('data_dir'), config.get('cloudy_models_name'))
else:
    CLOUDY_MODELS = os.path.join(os.path.join(ROOT_DIR, config.get('data_dir')), config.get('cloudy_models_name'))
if not os.path.isfile(CLOUDY_MODELS):
    log.warning("Pre-computed grid of Cloudy models ({}) is not found. "
                "Flux distribution in different lines will be unavailable".format(CLOUDY_MODELS))
    CLOUDY_MODELS = None

CLOUDY_SPEC_DEFAULTS = config.get('cloudy_default_params')

# path to the MAPPINGS models and some default parameters
if config.get('data_dir').startswith("."):
    MAPPINGS_MODELS = os.path.join(os.path.curdir, config.get('mappings_models_name'))
elif config.get('data_dir').startswith("/") or config.get('data_dir').startswith("\\"):
    MAPPINGS_MODELS = os.path.join(config.get('data_dir'), config.get('mappings_models_name'))
else:
    MAPPINGS_MODELS = os.path.join(os.path.join(ROOT_DIR, config.get('data_dir')),
                                   config.get('mappings_models_name'))
if not os.path.isfile(MAPPINGS_MODELS):
    log.warning("Pre-computed grid of Cloudy models ({}) is not found. "
                "Flux distribution in different lines will be unavailable".format(MAPPINGS_MODELS))
    MAPPINGS_MODELS = None
MAPPINGS_SPEC_DEFAULTS = config.get('mappings_default_params')


# path to the modeled continuum
if config.get('data_dir').startswith("."):
    CONTINUUM_MODELS = os.path.join(os.path.curdir, config.get('continuum_models_name'))
elif config.get('data_dir').startswith("/") or config.get('data_dir').startswith("\\"):
    CONTINUUM_MODELS = os.path.join(config.get('data_dir'), config.get('continuum_models_name'))
else:
    CONTINUUM_MODELS = os.path.join(os.path.join(ROOT_DIR, config.get('data_dir')), config.get('continuum_models_name'))
if not os.path.isfile(CONTINUUM_MODELS):
    log.warning("Pre-computed models of continuum are not available. ")
    CONTINUUM_MODELS = None

# path to the stellar library
if config.get('data_dir').startswith("."):
    STELLAR_LIBS = os.path.join(os.path.curdir, config.get('stellar_library_name'))
elif config.get('data_dir').startswith("/") or config.get('data_dir').startswith("\\"):
    STELLAR_LIBS = os.path.join(config.get('data_dir'), config.get('stellar_library_name'))
else:
    STELLAR_LIBS = os.path.join(os.path.join(ROOT_DIR, config.get('data_dir')), config.get('stellar_library_name'))
if not os.path.isfile(STELLAR_LIBS):
    log.warning("Stellar library is not found. It will be downloaded now")
    STELLAR_LIBS = None

# path to the directory where all computational results will be saved
if config.get('utah_cluster'):
    WORK_DIR = os.environ.get('LVM_SIMULATOR_OUT')
elif config.get('work_dir').startswith("/") or config.get('work_dir').startswith("\\") or \
        config.get('work_dir').startswith("."):
    WORK_DIR = config.get('work_dir')
else:
    WORK_DIR = os.path.join(ROOT_DIR, config.get('work_dir'))

# default properties of 2D simulator
COMMON_SETUP_2D = config.get('sim2d_params')

warnings.filterwarnings('ignore', r'divide by zero encountered in')
warnings.filterwarnings('ignore', r'invalid value encountered in ')

# package name should be pip package name
__version__ = get_package_version(path=__file__, package_name=NAME)

