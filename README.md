[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

# Information theoretical analysis of multivariate time series for the inference of higher-order interactions

## Project description

### Introduction

[See on Neurostars](https://neurostars.org/t/gsoc-2023-project-idea-5-1-optimization-of-the-computations-of-higher-order-interactions-and-integration-in-the-frites-python-package-175-h/24576):  Real-world systems are often characterized by higher-order interactions (HOIs) within multiplets i.e. groups of three or more units (Battiston et al., 2021). In neuroscience, most pieces of evidence we have about brain networks come from the interactions between pairs of brain regions, but little is known about what type of information remains hidden in the non-pairwise interactions. Interestingly, recent findings suggest that HOIs might be a better neural marker of neurodegeneration than standard pairwise approaches (Herzog et al., 2022).

Several methods have been proposed to estimate HOIs, from popular fields like graph- and information-theory. The HOI toolbox includes methods from information theory that are able to quantify higher-order behaviours from multivariate time series and infer interactions between variables beyond the pairwise level. The O-information (short name for “information about Organisational structure”) is an information-theoretical quantity to characterize statistical interdependencies within multiplets of three and more variables (Rosas et al., 2019). It allows us to not only quantify how much information multiplets of brain regions are carrying but also informs us on the nature of the information i.e. whether multiplets are carrying mainly redundant or synergistic information.

Estimating HOIs is computationally intensive. As an example, a cortical parcellation dividing the brain into 80 distinct regions involves estimating HOIs in 80.000 triplets, in 1.5 million quadruplets, in 24 million quintuplets, etc. The computational burden of the O-information only relies on simple quantities like entropies, which makes the O-information an ideal candidate to estimate HOIs in a reasonable time. Still, there is yet no neuroinformatic gold standard to estimate HOIs, in a decent amount of time and accessible to network enthusiasts encompassing experts and non-experts.

### Project aims and tasks

The repository was started as a GSoC 2023 Project Idea 5.1 focusing on the optimization of the computations of higher-order interactions and integration in the Frites python package. This project aims at optimizing the computations of dynamic HOIs. Ultimately, we want to be able to estimate the O-information on both simulated data and real brain data, possibly with high spatiotemporal resolution. To this end, we will start from an existing implementation we made during the BrainHack 2021 and 2022. We made two implementations of the HOIs: a first version using the standard NumPy library (Harris et al., 2020) that we compared with a second implementation using a more modern library called [Jax](https://github.com/google/jax) for accelerated linear algebra on both CPU and GPU. Finally, we will integrate the developments into the open-source toolbox called [Frites](https://github.com/brainets/frites) which currently only supports pairwise interactions.

We divided this project into five main tasks:

1. Optimize the low-level HOIs computations (70 hours): find ways to decrease computing time while keeping reasonable memory usage. Some ideas include faster entropy calculations, avoiding recomputing some quantities, parallel computations, etc.
2. Merge implementations into a single one (15 hours): merge the NumPy and Jax into a single function called “conn_hoi” and allow users to use the NumPy one without needing to install the Jax library (i.e. minimize requirements)
3. Data simulation (20 hours): add a function “simulate_hoi” to simulate HOIs
4. HOIs plotting (10 hours): create a function “plot_hoi” to plot the output of the “conn_hoi”. We will consider using either [XGI](https://github.com/ComplexGroupInteractions/xgi) or [HyperNetX](https://github.com/pnnl/HyperNetX)
5. Integrate the HOIs into the Frites software (60 hours): create a pull request (PR) to integrate the “conn_hoi” inside Frites. The PR will have to follow Frites’ formatting including input types and coding quality (pep8, flake8). We will also make an online comprehensive documentation accessible to non-experts, add unit tests and illustrative examples.

Ultimately, this project could lead to the establishment of a gold standard to go beyond pairwise interactions by measuring HOIs, accessible to Python experts such as to users with little programming knowledge.

## Description of the repo

This repository contains the Jax and Numpy implementations of the O-info (Rosas et al., 2019). As input types, we followed the structure of Frites and used `xarray.DataArray` which can seen as NumPy arrays with an external layer of labeling to improve data selection.

## Install the repo

To start working on the repo, clone it and run `pip install -e`. It should install the dependencies (frites, xarray etc.). Note that if you want to work on Jax, you will need to install it manually (`pip install --upgrade "jax[cpu]"` for the CPU version if you don't have a GPU).

## Resources

Please find important scientific resources in the `papers` folder:
- **Timme & Lapish, 2018 :** Introduction to information-theory in neuroscience
- **Ince et al., 2017 :** Presentation of the Gaussian-Copula Mutual information
- **Rosas et al., 2019:** Estimation of Higher-Order Interactions using the O-info
- **Combrisson et al., 2022 :** Presentation of the Python toolbox Frites


