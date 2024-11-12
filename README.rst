.. -*- mode: rst -*-

|Black|_ |Codecov|_ |JOSS|_

.. |Black| image:: https://img.shields.io/badge/code%20style-black-000000.svg
.. _Black: https://github.com/psf/black

.. |Codecov| image:: https://codecov.io/gh/brainets/hoi/graph/badge.svg?token=7PNM2VD994
.. _Codecov: https://codecov.io/gh/brainets/hoi

.. |JOSS| image:: https://joss.theoj.org/papers/10.21105/joss.07360/status.svg
.. _JOSS: https://doi.org/10.21105/joss.07360


.. image:: https://github.com/brainets/hoi/blob/main/docs/_static/hoi-logo.png
  :target: https://brainets.github.io/hoi/

Description
===========

HOI (Higher Order Interactions) is a Python package to go beyond pairwise interactions by quantifying the statistical dependencies between 2 or more units using information-theoretical metrics. The package is built on top of `Jax <https://github.com/google/jax>`_ allowing computations on CPU or GPU.

Installation
============

Dependencies
++++++++++++

HOI requires :

- Python (>= 3.8)
- numpy(>=1.22)
- scipy (>=1.9)
- jax
- pandas
- scikit-learn
- jax-tqdm
- tqdm

User installation
+++++++++++++++++

To install Jax on GPU or CPU-only, please refer to Jax's documentation : https://jax.readthedocs.io/en/latest/installation.html

If you already have a working installation of NumPy, SciPy and Jax,
the easiest way to install hoi is using ``pip``:

.. code-block:: shell

    pip install -U hoi

You can also install the latest version of the software directly from Github :

.. code-block:: shell

    pip install git+https://github.com/brainets/hoi.git


For developers
++++++++++++++

For developers, you can install it in develop mode with the following commands :

.. code-block:: shell

    git clone https://github.com/brainets/hoi.git
    cd hoi
    pip install -e .['full']

The full installation of HOI includes additional packages to test the software and build the documentation :

- pytest
- pytest-cov
- codecov
- xarray
- sphinx!=4.1.0
- sphinx-gallery
- pydata-sphinx-theme
- sphinxcontrib-bibtex
- numpydoc
- matplotlib
- flake8
- pep8-naming
- black


Help and Support
================

Documentation
+++++++++++++

- Link to the documentation: https://brainets.github.io/hoi/
- Overview of the mathematical background : https://brainets.github.io/hoi/theory.html
- List of implemented HOI metrics : https://brainets.github.io/hoi/api/modules.html
- Examples : https://brainets.github.io/hoi/auto_examples/index.html

Communication
+++++++++++++

For questions, please use the following link : https://github.com/brainets/hoi/discussions

Acknowledgments
===============

HOI was mainly developed during the Google Summer of Code 2023 (https://summerofcode.withgoogle.com/archive/2023/projects/z6hGpvLS)
