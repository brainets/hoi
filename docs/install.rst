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
- jax-tqdm
- tqdm

Here's the list of optional dependencies :

- xarray

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
