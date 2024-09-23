Quickstart
==========

HOI is a Python package to estimate :term:`Higher Order Interactions` from multivariate data. A network is composed of nodes (e.g. users in social network, brain areas in neuroscience, musicians in an orchestra etc.) and nodes are interacting together. Traditionally we measure pairwise interactions. HOI allows to go beyond the pairwise interactions by quantifying the interactions between 3, 4, ..., N nodes of the system. As we are using measures from the :term:`Information Theory`, we can further describe the type of interactions, i.e. whether nodes of the network tend to have redundant or synergistic interactions (see the definition of :term:`Redundancy`, :term:`Synergy`).

* **Installation :** to install HOI with its dependencies, see :ref:`installation`. If you are a developer or if you want to contribute to HOI, checkout the :ref:`contribute`.
* **Theoretical background :** For a detailed introduction to information theory and HOI, see :ref:`theory`. You can also have a look to our :ref:`glossary` to see the definition of the terms we are using here.
* **API and examples :** the list of functions and classes can be found in the section :ref:`hoi_modules`. For practical examples on how to use those functions, see :doc:`auto_examples/index`. For faster computations, HOI is built on top of Jax. Checkout the page :doc:`jax` for the performance claims.

Installation
++++++++++++

To install or update HOI, run the following command in your terminal :

.. code-block:: bash

   pip install -U hoi

Simulate data
+++++++++++++

We provide functions to simulate data and toy example. In a notebook or in a python script, you can run the following lines to simulate synergistic interactions between three variables :

.. code-block:: python

   from hoi.simulation import simulate_hoi_gauss

   data = simulate_hoi_gauss(n_samples=1000, triplet_character='synergy')

Compute Higher-Order Interactions
+++++++++++++++++++++++++++++++++

We provide a list of metrics of HOI (see :ref:`metrics`). Here, we are going to use the O-information (:class:`hoi.metrics.Oinfo`):

.. code-block:: python

   # import the O-information
   from hoi.metrics import Oinfo

   # define the model
   model = Oinfo(data)

   # compute hoi for multiplets with a minimum size of 3 and maximum size of 3
   # using the Gaussian Copula entropy
   hoi = model.fit(minsize=3, maxsize=3, method="gc")

Inspect the results
+++++++++++++++++++

To inspect your results, we provide a plotting function called :func:`hoi.plot.plot_landscape` to see how the information is spreading across orders together with :func:`hoi.utils.get_nbest_mult` to get a table of the multiplets with the strongest synergy or redundancy :


.. code-block:: python

   from hoi.plot import plot_landscape
   from hoi.utils import get_nbest_mult

   # plot the landscape
   plot_landscape(hoi, model=model)

   # print the summary table
   print(get_nbest_mult(hoi, model=model))


Practical recommendations
+++++++++++++++++++++++++

Robust estimations of HOI strongly rely on the accuity of measuring entropy/mutual information on/between (potentially highly) multivariate data. In the :doc:`auto_examples/index` section you can find benchmarks of our entropy estimators. Here we recommend :

* **Measuring entropy and mutual information :** we recommend the Gaussian Copula method (`method="gc"`). Although this measure is not accurate for capturing relationships beyond the gaussian assumption (see :ref:`sphx_glr_auto_examples_it_plot_entropies.py`), this method performs relatively well for multivariate data (see :ref:`sphx_glr_auto_examples_it_plot_entropies_mvar.py`)
* **Measuring Higher-Order Interactions for network behavior and network encoding :** for network behavior and ncoding, we recommend respectively the O-information :class:`hoi.metrics.Oinfo` and the :class:`hoi.metrics.GradientOinfo`. Although both metrics suffer from the same limitations, like the spreading to higher orders, this can be mitigated using a boostrap approach (see :ref:`sphx_glr_auto_examples_statistics_plot_bootstrapping.py`). Otherwise, both metrics are usually pretty accurate to retrieve the type of interactions between variables, especially once combined with the Gaussian Copula.


Other softwares for the analysis of higher-order interactions
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

Please find bellow a list of additional softwares for the analysis of higher-order interactions :

- `XGI <https://xgi.readthedocs.io/>`_ : Python software for modeling, analyzing, and visualizing higher-order interactions
- `NetworkX <https://networkx.org/>`_ : Python package for the creation, manipulation, and study of the structure, dynamics, and functions of complex networks
- `TopoNetX <https://pyt-team.github.io/toponetx/>`_ : Python package for computing on topological domains
- `HGX <https://hypergraphx.readthedocs.io>`_ : Python library for higher-order network analysis
- `InfoTopo <https://infotopo.readthedocs.io/>`_ : original Python implementation of the :class:`hoi.metrics.InfoTopo` estimator
- `infotheory <http://mcandadai.com/infotheory/>`_ : C++, and usable in Python as well, Infotheory is a software to perform information theoretic analysis on multivariate data
- `dit <https://dit.readthedocs.io>`_ : Python package for discrete information theory
- `IDTxl <https://pwollstadt.github.io/IDTxl/html/index.html>`_ : Python software for efficient inference of networks and their node dynamics from multivariate time series data using information theory
- `pyphi <https://github.com/wmayner/pyphi>`_ : Python library for computing integrated information
