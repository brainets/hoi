.. -*- mode: rst -*-

|Black|_ |Codecov|_

.. |Black| image:: https://img.shields.io/badge/code%20style-black-000000.svg
.. _Black: https://github.com/psf/black

.. |Codecov| image:: https://codecov.io/gh/brainets/hoi/graph/badge.svg?token=7PNM2VD994
.. _Codecov: https://codecov.io/gh/brainets/hoi


HOI jax implementation
======================

Description
+++++++++++

**HOI** (Higher Order Interactions) is a Python package to go beyond pairwise interactions by quantifying the statistical dependencies between 2 or more units using information-theoretical metrics. The package is built on top of `Jax <https://github.com/google/jax>`_ allowing computations on CPU or GPU.

.. grid:: 3

    .. grid-item-card:: :material-regular:`rocket_launch;2em` Install hoi
      :columns: 12 6 6 4
      :link: installation
      :link-type: ref
      :class-card: box1

    .. grid-item-card:: :material-regular:`library_books;2em` List of functions
      :columns: 12 6 6 4
      :link: hoi_modules
      :link-type: ref
      :class-card: box2

    .. grid-item-card:: :material-regular:`laptop_chromebook;2em` Examples
      :columns: 12 6 6 4
      :link: auto_examples/index
      :link-type: doc
      :class-card: box3

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Getting Started 🚀

   Installation <install>
   Theoretical background <overview/index>


.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Further Resources 🔪

   List of functions <api/modules>
   Examples <auto_examples/index>
