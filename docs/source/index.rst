KINTERA Documentation
=====================

Welcome to KINTERA's documentation!

KINTERA is a high-performance library for atmospheric chemistry and thermodynamics calculations, combining C++ performance with Python accessibility through pybind11 bindings.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   quickstart
   user_guide
   api_reference
   examples

Overview
--------

KINTERA provides efficient implementations of:

* **Chemical kinetics calculations** - Comprehensive reaction mechanism support
* **Thermodynamic equation of state** - Advanced phase equilibrium computations
* **Atmospheric chemistry models** - Cloud physics with nucleation and condensation modeling
* **PyTorch Integration** - Native tensor operations with GPU acceleration support

Key Features
------------

* **High Performance**: C++17 core with optional CUDA support
* **Python Interface**: Full Python API via pybind11
* **PyTorch Integration**: Native tensor operations using PyTorch
* **Type Hints**: Complete type annotations for IDE support and type checking
* **Flexible**: Support for Earth, Jupiter, and custom atmospheric compositions

Getting Started
---------------

Install KINTERA with pip:

.. code-block:: bash

   pip install numpy 'torch==2.7.1' 'pyharp>=1.7.1'
   pip install kintera

For more detailed installation instructions, see the :doc:`installation` guide.

Quick Example
-------------

Here's a simple example to get you started:

.. code-block:: python

   import torch
   import kintera
   from kintera import ThermoOptions, ThermoX

   # Load configuration from YAML
   op = ThermoOptions.from_yaml("jupiter.yaml")
   thermo = ThermoX(op)

   # Set up state variables
   temp = torch.tensor([200.], dtype=torch.float64)
   pres = torch.tensor([1.e5], dtype=torch.float64)

   # Define species composition
   species = op.species()
   nspecies = len(species)
   xfrac = torch.rand((1, 1, nspecies), dtype=torch.float64)
   xfrac /= xfrac.sum(dim=-1, keepdim=True)

   # Compute equilibrium
   thermo.forward(temp, pres, xfrac)
   print("Equilibrium composition:", xfrac)

See the :doc:`quickstart` guide for more examples.

Support and Contributing
------------------------

* **Issues**: `GitHub Issues <https://github.com/chengcli/kintera/issues>`_
* **Source Code**: `GitHub Repository <https://github.com/chengcli/kintera>`_
* **Contact**: chengcli@umich.edu

License
-------

See the `LICENSE <https://github.com/chengcli/kintera/blob/main/LICENSE>`_ file for details.

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
