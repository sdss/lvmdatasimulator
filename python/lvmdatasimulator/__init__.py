# encoding: utf-8
import os
from sdsstools import get_config, get_logger, get_package_version
import warnings
from urllib.request import urlretrieve
from tqdm import tqdm

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
large_files_location = config.get('large_files_location')

if config.get('data_dir').startswith("."):
    DATA_DIR = os.path.curdir
elif config.get('data_dir').startswith("/") or config.get('data_dir').startswith("\\"):
    DATA_DIR = config.get('data_dir')
else:
    DATA_DIR = os.path.join(ROOT_DIR, config.get('data_dir'))

# path to the Cloudy models and some default parameters
CLOUDY_MODELS = os.path.join(DATA_DIR, config.get('cloudy_models_name'))
if not os.path.isfile(CLOUDY_MODELS):
    log.warning("Pre-computed grid of Cloudy models ({}) is not found. "
                "Flux distribution in different lines will be unavailable".format(CLOUDY_MODELS))
    CLOUDY_MODELS = None

CLOUDY_SPEC_DEFAULTS = config.get('cloudy_default_params')

# path to the MAPPINGS models and some default parameters
MAPPINGS_MODELS = os.path.join(DATA_DIR, config.get('mappings_models_name'))
if not os.path.isfile(MAPPINGS_MODELS):
    log.warning("Pre-computed grid of Cloudy models ({}) is not found. "
                "Flux distribution in different lines will be unavailable".format(MAPPINGS_MODELS))
    MAPPINGS_MODELS = None
MAPPINGS_SPEC_DEFAULTS = config.get('mappings_default_params')


# path to the modeled continuum
CONTINUUM_MODELS = os.path.join(DATA_DIR, config.get('continuum_models_name'))
if not os.path.isfile(CONTINUUM_MODELS):
    log.warning("Pre-computed models of continuum are not available. ")
    CONTINUUM_MODELS = None

# path to the stellar library
STELLAR_LIBS = os.path.join(DATA_DIR, config.get('stellar_library_name'))
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


def my_hook(t):
    """Wraps tqdm instance.
    Don't forget to close() or __exit__()
    the tqdm instance once you're done with it (easiest using `with` syntax).
    Example
    -------
    >>> with tqdm(...) as t:
    ...     reporthook = my_hook(t)
    ...     urllib.urlretrieve(..., reporthook=reporthook)

    this code comes from https://gist.github.com/leimao/37ff6e990b3226c2c9670a2cd1e4a6f5
    """
    last_b = [0]

    def update_to(b=1, bsize=1, tsize=None):
        """
        b  : int, optional
            Number of blocks transferred so far [default: 1].
        bsize  : int, optional
            Size of each block (in tqdm units) [default: 1].
        tsize  : int, optional
            Total size (in tqdm units). If [default: None] remains unchanged.
        """
        if tsize is not None:
            t.total = tsize
        t.update((b - last_b[0]) * bsize)
        last_b[0] = b

    return update_to


def download_data():

    log.info("Downloading new data from SAS")
    # cloudy models
    remote_path_cloudy = large_files_location+config.get('cloudy_models_name')
    CLOUDY_MODELS = os.path.join(DATA_DIR, config.get('cloudy_models_name'))

    with tqdm(unit = 'B', unit_scale = True, unit_divisor = 1024, miniters = 1) as t:
        urlretrieve(remote_path_cloudy, CLOUDY_MODELS, my_hook(t))
    log.info(f"{config.get('cloudy_models_name')} downloaded")

    remote_path_stars = large_files_location+config.get('stellar_library_name')
    STELLAR_LIBS = os.path.join(DATA_DIR, config.get('stellar_library_name'))

    with tqdm(unit = 'B', unit_scale = True, unit_divisor = 1024, miniters = 1) as t:
        urlretrieve(remote_path_stars, STELLAR_LIBS, my_hook(t))
    log.info(f"{config.get('stellar_library_name')} downloaded")

    for cam in ['blue', 'red', 'ir']:
        for i in range(3):
            focus_name = f'focus_lvm_{cam}{i+1}.fits.gz'
            remote_path_focus = large_files_location+focus_name
            out_path =os.path.join(DATA_DIR, 'focus', focus_name)
            with tqdm(unit = 'B', unit_scale = True, unit_divisor = 1024, miniters = 1) as t:
                urlretrieve(remote_path_focus, out_path, my_hook(t))
            log.info(f"{focus_name} downloaded")