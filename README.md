# lvmdatasimulator

![Versions](https://img.shields.io/badge/python->3.7-blue)
[![Documentation Status](https://readthedocs.org/projects/sdss-lvmdatasimulator/badge/?version=latest)](https://sdss-lvmdatasimulator.readthedocs.io/en/latest/?badge=latest)
[![Travis (.org)](https://img.shields.io/travis/sdss/lvmdatasimulator)](https://travis-ci.org/sdss/lvmdatasimulator)
[![codecov](https://codecov.io/gh/sdss/lvmdatasimulator/branch/main/graph/badge.svg)](https://codecov.io/gh/sdss/lvmdatasimulator)

Simulator of LVM data for testing DRP and DAP



### Major limitations of the current version:

- Interstellar extinction from the produced dark clouds:  
  - Works for ISM, but not for background stars
  - In the output maps and spectra it is based on the mean value of Av in the fiber, 
  not accounts for the density variations within a fiber 
- Line-of-sight velocities:
  - Systemic velocities, velocity fields and 
  variations of the line profiles are accounted for the extraction of the
  emission line spectra, but **not for the continuum**.
  - This is also true for the extraction of the input map.
- Interpolation to the instrument wavelength grid:
  - In the fast-mode it is performed in not truly flux-conserving regime (precision ~10%)
  - LSF of the instrument is not taken into account
- Observational conditions and parameters:
  - PSF is not taken into account


