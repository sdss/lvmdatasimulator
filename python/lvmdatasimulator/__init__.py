# encoding: utf-8

from sdsstools import get_config, get_logger, get_package_version


# pip package name
NAME = 'sdss-lvmdatasimulator'

# Loads config. config name is the package name.
config = get_config('lvmdatasimulator')

# Inits the logging system as NAME. Only shell logging, and exception and warning catching.
# File logging can be started by calling log.start_file_logger(path).  Filename can be different
# than NAME.
log = get_logger(NAME)


# package name should be pip package name
__version__ = get_package_version(path=__file__, package_name=NAME)
