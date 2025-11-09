Installation
============

This guide covers how to install KINTERA on various platforms.

Prerequisites
-------------

System Requirements
~~~~~~~~~~~~~~~~~~~

* **C++ Compiler**: Support for C++17 (GCC 7+, Clang 5+, or MSVC 2017+)
* **CMake**: Version 3.18 or higher
* **Python**: Version 3.9 or higher
* **NetCDF**: NetCDF C library

Python Dependencies
~~~~~~~~~~~~~~~~~~~

Required Python packages:

* ``numpy``
* ``torch`` (version 2.7.0-2.7.1)
* ``pyharp`` (version 1.7.2+)
* ``pytest`` (for testing)

Platform-Specific Setup
-----------------------

Linux (Ubuntu/Debian)
~~~~~~~~~~~~~~~~~~~~~

Install system dependencies:

.. code-block:: bash

   sudo apt-get update
   sudo apt-get install -y build-essential cmake libnetcdf-dev

macOS
~~~~~

Install dependencies using Homebrew:

.. code-block:: bash

   brew update
   brew install cmake netcdf

Installation Methods
--------------------

Quick Installation (PyPI)
~~~~~~~~~~~~~~~~~~~~~~~~~

The simplest way to install KINTERA:

.. code-block:: bash

   # Install Python dependencies
   pip install numpy 'torch==2.7.1' 'pyharp>=1.7.1'
   
   # Install KINTERA
   pip install kintera

Installation from Source
~~~~~~~~~~~~~~~~~~~~~~~~

For the latest development version or to build with custom options:

1. **Clone the repository**:

   .. code-block:: bash

      git clone https://github.com/chengcli/kintera.git
      cd kintera

2. **Install Python dependencies**:

   .. code-block:: bash

      pip install numpy 'torch==2.7.1' 'pyharp>=1.7.1'

3. **Configure and build the C++ library**:

   .. code-block:: bash

      cmake -B build
      cmake --build build --parallel

4. **Install the Python package**:

   .. code-block:: bash

      pip install .

Building with CUDA Support
~~~~~~~~~~~~~~~~~~~~~~~~~~~

To build with CUDA support for GPU acceleration:

.. code-block:: bash

   cmake -B build -DCUDA_ENABLE=ON
   cmake --build build --parallel
   pip install .

Verifying Installation
----------------------

Test your installation:

.. code-block:: python

   import kintera
   print(kintera.__version__)
   
   # Check available classes
   from kintera import ThermoOptions, ThermoX, SpeciesThermo
   print("KINTERA installed successfully!")

Running Tests
~~~~~~~~~~~~~

If you installed from source, you can run the test suite:

.. code-block:: bash

   cd build/tests
   ctest

Development Installation
------------------------

For development, install in editable mode with pre-commit hooks:

.. code-block:: bash

   # Clone and navigate to repository
   git clone https://github.com/chengcli/kintera.git
   cd kintera
   
   # Install dependencies
   pip install numpy 'torch==2.7.1' 'pyharp>=1.7.1'
   pip install pre-commit pytest
   
   # Build C++ library
   cmake -B build
   cmake --build build --parallel
   
   # Install Python package in editable mode
   pip install -e .
   
   # Set up pre-commit hooks
   pre-commit install

Dependency Cache
----------------

KINTERA uses a dependency cache to speed up builds. After the first successful build:

* Cache files are stored in the ``.cache`` directory
* Subsequent builds can run offline
* To force a clean rebuild:

  .. code-block:: bash

     rm -rf .cache build

Troubleshooting
---------------

Import Errors
~~~~~~~~~~~~~

If you encounter import errors, ensure:

1. Python dependencies are installed correctly
2. The C++ library was built successfully
3. You're using Python 3.9 or higher

Build Failures
~~~~~~~~~~~~~~

For build issues:

1. Ensure CMake version is 3.18 or higher: ``cmake --version``
2. Check that you have a C++17-compatible compiler
3. Verify NetCDF is installed: ``nc-config --version``

For more help, visit the `GitHub Issues <https://github.com/chengcli/kintera/issues>`_ page.
