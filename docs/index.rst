.. -*- mode: rst -*-

|Black|_ |Codecov|_

.. |Black| image:: https://img.shields.io/badge/code%20style-black-000000.svg
.. _Black: https://github.com/psf/black

.. |Codecov| image:: https://codecov.io/gh/brainets/hoi/graph/badge.svg?token=7PNM2VD994
.. _Codecov: https://codecov.io/gh/brainets/hoi


HOI : High-Performance Estimation of Higher-Order Interactions
==============================================================

**HOI** (Higher Order Interactions) is a Python package to go beyond pairwise interactions by quantifying the statistical dependencies between 2 or more units using information-theoretical metrics. The package is built on top of `Jax <https://github.com/google/jax>`_ allowing computations on CPU or GPU.

.. grid:: 3
   :margin: 0
   :padding: 0
   :gutter: 0

   .. grid-item-card:: Familiar API
      :columns: 12 6 6 4
      :class-card: sd-border-0
      :shadow: None

      HOI provides a familiar Scikit-learn style API for ease of adoption by researchers and engineers.

   .. grid-item-card:: Metrics
      :columns: 12 6 6 4
      :class-card: sd-border-0
      :shadow: None

      HOI provides cutting-edge and most up-to-date metrics to estimate higher-order interactions

   .. grid-item-card:: Run Anywhere
      :columns: 12 6 6 4
      :class-card: sd-border-0
      :shadow: None

      The same code executes on multiple backends, including CPU, GPU, & TPU

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

Get involved
++++++++++++

If you want to contribute to HOI, please checkout the :ref:`contribute`

.. How to Cite
.. +++++++++++

.. If you use HOI in your data analysis, please use the following citations in your publications :

.. .. code-block:: latex

..    bibtex entry

Funding
+++++++

The HOI package has been supported by the `Google Summer of Code 2023 <https://summerofcode.withgoogle.com/archive/2023/projects/z6hGpvLS>`_. We also acknowledge the A*Midex Foundation of Aix-Marseille University project “Hinteract” (AMX-22-RE-AB-071) and the EU’s Horizon 2020 Framework Programme for Research and Innovation under the Specific Grant Agreements No. 101147319 (EBRAINS 2.0 Project).

License
+++++++

This project is licensed under the `BSD 3-Clause License <https://github.com/brainets/hoi/blob/main/LICENSE>`_.


.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Getting Started 🚀

   Installation <install>
   Quickstart <quickstart>
   Examples <auto_examples/index>
   Glossary <glossary>
   Theoretical background <theory>


.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Further Resources 🔪

   Public API: list of functions <api/modules>
   Jax <jax>
   Developer Documentation <contributor_guide>
   List of contributors <contributors>
