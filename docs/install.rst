:orphan:

.. _installation:

Installation
------------


Dependencies
++++++++++++

HOI requires :

- Python (>= 3.8)
- numpy(>=1.22)
- scipy (>=1.9)
- jax
- pandas
- scikit-learn
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

