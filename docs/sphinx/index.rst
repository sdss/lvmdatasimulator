
lvmdatasimulator's documentation
=============================================

The `lvmdatasimulator` is a Python package developed to simulate the data produced by the SDSS-V Local Volume Mapper (LVM).
It can simulate realistic raw frames, that can be reduced using the official LVM data reduction package, or a simplified
version of the reduced data, ready to be analized using the official LVM data analysis pipeline or any custom data
analysis tool.

Installation
------------


To date, the `lvmdatasimulator` is available only through the official SDSS github. To obtain the package, please clone
the repository and install it in developement mode via:

.. code-block:: console

  $ git clone --recursive https://github.com/sdss/lvmdatasimulator --branch main
  $ python -m pip install -e .

To work correctly, the simulator also needs a few libraries that are not provided within the GitHub repository.
Therefore, before running the simulator please download the following files:

.. code-block:: console

  $ wget https://data.sdss5.org/resources/lvmdatasimulator/data/for_download/LVM_cloudy_models_phys.fits
  $ wget https://data.sdss5.org/resources/lvmdatasimulator/data/for_download/pollux_resampled_v0.fits

The downloaded files should be moved to the `lvmdatasimulator\data` directory.

More detailed instructions for the installation of the package, and some basic information on how
to run the simulator can be found in :ref:`Getting Started<getting_started>`.


Contents
--------

.. toctree::
  :maxdepth: 1

  What's new in lvmdatasimulator? <CHANGELOG>
  Introduction to lvmdatasimulator <intro>


Reference
---------

.. toctree::
   :maxdepth: 1

   api


Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
