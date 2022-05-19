# lvmdatasimulator

![Versions](https://img.shields.io/badge/python->3.7-blue)
[![Documentation Status](https://readthedocs.org/projects/sdss-lvmdatasimulator/badge/?version=latest)](https://sdss-lvmdatasimulator.readthedocs.io/en/latest/?badge=latest)
[![Travis (.org)](https://img.shields.io/travis/sdss/lvmdatasimulator)](https://travis-ci.org/sdss/lvmdatasimulator)
[![codecov](https://codecov.io/gh/sdss/lvmdatasimulator/branch/main/graph/badge.svg)](https://codecov.io/gh/sdss/lvmdatasimulator)

Simulator of LVM data for testing DRP and DAP. It can be used also as an advanced exposure time calculator for LVM.

**In the current, early, version only reduced data with a very simplified instrument setup can be simulated.**

## Instalation and configuration
To install the package, please run

```git clone --recursive https://github.com/sdss/lvmdatasimulator```

Don't forget to download all necessary files with the pre-computed grids of models. 
The details are given in this [notebook](examples/LVM_data_simulator_tutorial.ipynb)


## How to use the code

Shortly, the current version of the code has three


### Major limitations of the current version:


- Source field:
  - Nebular continuum is not present in the existing Cloudy models; accounting for it is not implemented yet
  - Pre-generated shocks models are absent
  - Central ionizing source is not produced in the output. 
  If needed, it is possible to add the star with corresponding parameters. 
  - Addition of the stars overwrite the previously fetched or generated starlist
- Interstellar extinction from the produced dark clouds:  
  - Works for ISM, but not for background stars
  - In the output maps and spectra it is based on the mean value of Av in the fiber, not accounting for the density variations within a fiber
- Line-of-sight velocities:
  - Systemic velocities, velocity fields and variations of the line profiles are accounted for the extraction of the emission line spectra, but **not for the continuum**.
  - This is also true for the extraction of the input map.
- Simplified wavelength solution:
  - A single, uniformly sampled wavelength array covering the full LVM wavelength range is used.
  - A Gaussian line spread function is considered. Its width is constant across the full wavelength range.
- Interpolation to the instrument wavelength grid:
  - Running the simulator with **fast=True** means that a simple interpolation is used to regrid the spectra. On average the flux is conserved, but the shape of the line can be affected. Set **fast=False** if this is a problem.
  - Resampling with **fast=False** uses a precise flux conserving algorithm for the resampling. Parallelization is used to speed-up the simulation, but it is still significantly slower than using **fast=True**.
- Observational conditions and parameters:
  - PSF is not implemented yet
  - Atmospheric dispersion is not implemented yet
  - Sky transparency is not implemented yet

