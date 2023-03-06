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
The details are given in this [notebook](examples/LVM_data_simulator_tutorial1.ipynb) and in this
[section](#files)

**N.B.:** We have been notified that there are problems installing the package with the M1 MacBook. The solution seems to be installing the **hdf5** package. Including this package in the requirements, however, creates issues with the installation on other platforms (e.g. Ubuntu). So, **if you are trying to install this package on a new Mac, please first do ``pip install hdf5``**.


## How to use the code

Shortly, the current version of the code can be divided three main sections:
- **Source field**, which produces the 2 fits files containing the stellar and ISM components that are used further for simulations.
- **Instrumentation and observing conditions** -with this part of the code the user can configure the properties of the LVM instrument (e.g., number of the fibers to use in simulations, their location etc.), of the sky conditions (e.g. Sky conditions, airmass, etc ) and the details of the observations (e.g. exposure time)
- **Simulator** join all these together and produce the output "observed" spectra and images from the generated "source field"
- **Simulator2D** is an alternative to **Simulator** for making the Raw LVM spectra (mimics DRP input)

The detailed documentation of these blocks will be added later to this readme.
At the moment, we prepared two tutorials for users on how to use the code.

- [Tutorial #1](https://github.com/sdss/lvmdatasimulator/blob/main/examples/LVM_data_simulator_tutorial1.ipynb) describes the necessary steps to properly install and configure the package and overview the very basic steps to run the simulations.
- [Tutorial #2](https://github.com/sdss/lvmdatasimulator/blob/main/examples/LVM_data_simulator_tutorial2.ipynb) digs deeper into the configuration of some aspects of the simulator (fiber bundle, Observation class) and describe the properties of the output files.
- [Tutorial #3](https://github.com/sdss/lvmdatasimulator/blob/main/examples/LVM_data_simulator_tutorial3.ipynb) explains in more details how to configure the source field, simulate different types of nebulae. It has two major parts. One describes the very simplified simulations, while another considers different aspects on how to deal with the varying line rations and gas kinematics.
- [Tutorial #4](https://github.com/sdss/lvmdatasimulator/blob/main/examples/LVM_data_simulator_tutorial4.ipynb) shows how to use the simulator as a simple ETC.
- [Tutorial #5](https://github.com/sdss/lvmdatasimulator/blob/main/examples/LVM_data_simulator_tutorial5.ipynb) describes 2D simulations (how to produce Raw LVM spectra = DRP input)



### Major limitations of the current version:


- Source field:
  - Nebular continuum is not present in the existing Cloudy models; accounting for it is not implemented yet
  - Pre-generated shocks models are absent
  - Central ionizing source is not produced in the output.
  If needed, it is possible to add the star with corresponding parameters.
- Interstellar extinction from the produced dark clouds:
  - Works for ISM, but not for background stars
  - In the output maps and spectra it is based on the mean value of Av in the fiber, not accounting for the density variations within a fiber
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

### Known bugs or missing features that will be fixed soon
- Addition of the stars overwrite the previously fetched or generated starlist
- Line-of-sight velocities:
  - Systemic velocities, velocity fields and variations of the line profiles are accounted for the extraction of the emission line spectra, but **not for the continuum**.
  - This is also true for the extraction of the input map.

## Desrciption of the required and generated files
<a id="files"></a>

Source field require the pre-calculated Cloudy models for proper work. Default version can be obtained in this way:
``wget ftp@ftp.das.uchile.cl:/pub/enrico/LVM_cloudy_models_phys.fits``
It should be stored in the ``data`` directory. This file has a specific format that is described below. Users can generate their own model grids.

Another file that is required for running source field is the templates of stellar population. It can be fetched in this way:
``wget ftp@ftp.das.uchile.cl:/pub/enrico/pollux_resampled_v0.fits``

Finally, the Starburst99 models are useful to produce the continuum. They are already in the data directory (``LVM_continuum_models.fits``)

### Adopted format of the pre-generated Cloudy models.
In lvmdatasimulator we use the specific format of the Cloudy models that is described below. If someone wants to run simulations using their own model grids, they can save such grids in the same format and rewrite the path to the model grid in the config file

- The file is multi-extensional. Last extension (*Summary*) contains the BINTABLE summarizing the information about every model
- Every other extension is Image and has name = *Model_ID* as it is defined in the *Summary* table. All main properties of the model are listed as fields in the header
- First column of each model defines the exact wavelength of the emission lines (starts from the second row)
- Second column has the normalized **integrated fluxes over the nebula** (starts from the second row). They are used when the line ratios don't vary across the nebula.
- All other columns correspond to the normalized **fluxes at each given radii** in the nebula  (starts from the second row). This is used only for Cloud or Bubble when the line ratios vary across the nebula.
- First row define the position within the cloud (as a fraction from r_in to r_out), starts from third column.


## Produced fits-files with information on the ISM:

The ISM components generated by the lvmdatasimulator (source field) is saved to a multi-extensions fits file.
It is possible to externally modify the file before running simulator. Every nebula in this file has their own ID.

The following extension can be included in the file:
- Primary extension contains the main parameters of the ISM object (size, wcs, etc.). It also shows the combined image of nebulae in specific wavelength range (by default: 6560-6565A)
- *Comp_{ID}\_Brightness* - distribution of the maximal brightness (by default - in Halpha).
In case of the dark nebula - maximal extinction. Brightness is in erg/s/cm^2 per arcsec^2, extinction - in mags per arcsec^2
- *Comp_{ID}\_Fluxratios* - Fluxes of the emission lines before the convolution with the brightness and velocity distribution. This is used if the spectrum doesn't change accross the nebula.
It is an array wit size n_lines x 2 elements, where the first row shows the exact wavelength of the line, and the second row - its flux (normalized to Halpha or any other line corresponding to *Comp_{ID}\_Brightness*).
- *Comp_{ID}\_Flux_{line}* - distribution of the brightness in the particular line. *line* is equal to the exact wavelength (e.g., '6562.81'). Not used if the spectrum doesn't change across the nebula. Brightness is per arcsec^2
- *Comp_{ID}\_Vel* - map of the LOS velocity; if not present, then velocity is assumed to be uniform (= *VSYS*) for each pixel.
- *Comp_{ID}\_Disp* - map of the distribution of velocity dispersion in this particular nebula; if not present, then assumed to be uniform and equal to *TURBVEL*
- *Comp_{ID}\_LineProfile* - 3D array with the line profile relative to SysVel; **if present - it already includes broadening and variations in velocity field !**
- *Comp_{ID}\_Continuum* - 2D array containing the information about the wavelength (1 row) and flux (second row) of the continuum
- *Comp_{ID}\_PhysParams* - 3xN array containing the radial distribution of the physical parameters: Te (2nd row) and Ne (3rd row). First row contains radius from the nebula center in parsecs. Created only if the nebula is Cloud, Bubble, Cloud3D or Bubble3d and corresponding spectrum is from Cloudy grid



The primary extension contains the following important fields:
  - WCS defining the simulated FOV
  - Width - total X-size of the FOV, in px
  - Height - total Y-size of the FOV, in px
  - PhysRes - pixel size for all ISM content, in pc/px
  - Dist - distance to the ISM (to be used by default for those nebulae without explicitly mentioned distance)
  - Vsys - systemic velocity of the ISM (to be used by default for those nebulae without explicitly mentioned SysVel)
  - TurbSig - velocity dispersion of the ISM (to be used by default for those nebulae without explicitly mentioned TurbVel)
  - NObj - total number of nebulae contained in the file
  - IM_WL - Rest-frame wavelength range for the reference image in the primary extension (default: 6560-6565A = Halpha line + underlying continuum)

The headers of each other extension can contain the following fields:
  - Nebtype = Type of the current nebula, can be: DIG, Filament, Bubble, Cloud, Rectangle, Circle, Ellipse, Galaxy, CustomNebula
  - Dark = 1 if it is a dark nebula (with extinction, no emission), 0 if it is an emission nebula
  - X0, Y0 = position of the bottom-left corner in the full field of view
  - Zorder = 0: all nebulae structures will be added to the final cube according to Zorder. DIG has Zorder=-1,
  all other nebulae => Zorder = 0, but this can be changed manualy. If the Nebula has Zorder > than for another one,
  it means that it is observed at the foreground (and thus the extinction of the latter does matter)
  - Radius - radius of the nebula in pc (for Cloud, Bubble, Ellipse (= major semi-axis), Circle)
  - Reff - effective radius of the nebula, in pc (for Galaxy)
  - AxRat - minor-to-major axes ratio for Ellipse or Galaxy
  - Height - Y-size of the nebula, in pc
  - Width - Y-size of the nebula, or width of the Filament, in pc
  - Length - length of the Filament, in pc
  - PA - positional angle of the nebula, degrees (for Ellipse, Galaxy, Filament)
  - PertScl - Scale of the random perturbations, pc
  - PertOrd - order of generated perturbations, if appropriate (spherical harmonics in case of Bubble, Cloud)
  - PertAmp - Max amplitude of random perturbations
  - MaxBrt - maximal brightness in Halpha line, erg/s/cm^2/arcsec^2
  - MaxExt - maximal extinction in V, mag/arcsec^2
  - Vexp - expansion velocity, km/s. 0 for all but Bubble
  - SysVel - systemic velocity, km/s
  - TurbVel - velocity dispersion (e.g., due to turbulence) in the nebula, km/s
  - VGrad - velocity gradient, km/s per pc
  - VRot - Rotational velocity, km/s (for Galaxy)
  - PAKin - PA of kinematical axis, degrees (to define the direction of VGrad or VRot)
  - SpecID - id of the pregenerated Cloudy model used for this spectrum
  - LineRat - 'Constant' or 'Variable' - define if the lines fluxes ratios change across the nebula
  - Nlines - if > 0, then only brightest Nlines lines will be used for the spectra extraction
  - Distance - assumed distance to the current in kpc
  - NChunks - how many chunks will be used to extract the spectra of this nebula by simulator (by default, -1 that means "to be defined automatically")
  - ContType - type of the continuum saved for this nebula ('model', or 'BB', or 'poly')
  - ContWl - reference wavelength (if float) or band (if string, can be B, V, R, I, sdss-g, sdss-r, sdss-i) that is used for continuum normalization
  - ContFlux - Continuum brightness (in erg/s/cm^2/asec^2/AA) at the reference wavelength or band
  - ContMag - Continuum magnitude (in mag/asec^2) at the reference wavelength or band; skipped if ContFlux is not zero
  - Lambda = Exact wavelength of the current line (if the extension contains the flux in a single line)

