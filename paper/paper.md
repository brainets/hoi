---
title:  'High-performance estimation of Higher-Order Interactions from multivariate data'
tags:
  - python
  - higher-order interactions
  - CPU/GPU
  - information theory
authors:
  - name: Matteo Neri
    orcid: 0009-0007-0998-552X
    affiliation: 1
    equal-contrib: true
  - name: Dishie Vinchhi
    affiliation: 3
    equal-contrib: true
  - name: Christian Ferreyra
    affiliation: '1,5'
  - name: Thomas Robiglio
    affiliation: 4
  - name: Onur Ates
    affiliation: 1
  - name: Andrea Brovelli
    orcid: 0000-0002-5342-1330
    affiliation: 1
  - name: Marlis Ontivero-Ortega
    orcid: 0000-0003-0084-8274
    affiliation: 2
  - name: Daniele Marinazzo
    orcid: 0000-0002-9803-0122
    affiliation: 2
    equal-contrib: true
  - name: Etienne Combrisson
    orcid: 0000-0002-7362-3247
    affiliation: 1
    equal-contrib: true
affiliations:
 - name: Institut de Neurosciences de la Timone, Aix Marseille Université, UMR 7289 CNRS, 13005, Marseille, France
   index: 1
 - name: University of Ghent, Ghent, Belgium
   index: 2
 - name: Veermata Jijabai Technological Institute, Mumbai
   index: 3
 - name: Department of Network and Data Science, Central European University, Vienna, Austria
   index: 4
 - name:  Laboratoire d’Informatique et des Systèmes, Aix Marseille Université, UMR 7020 CNRS, Marseille, France
   index: 5
date: 19 September 2024
bibliography: paper.bib

---

# Summary

[`HOI`](https://brainets.github.io/hoi/) (Higher-Order Interactions) is a Python toolbox to measure higher-order information theoretic metrics from multivariate data. Higher-order interactions refer to interactions that go beyond pairwise connections between nodes in a network [@battiston:2021; @baudot:2019; @rosas:2019; @luppi:2024; @herzog:2022; @gatica:2021]. The `HOI` toolbox provides easy-to-use information theoretical metrics to estimate pairwise and higher-order information from multivariate data. The toolbox contains cutting-edge methods, along with core entropy and mutual information functions, which serve as building blocks for all metrics. In this way, `HOI` is accessible both to scientists with basic Python knowledge using pre-implemented functions and to experts who wish to develop new metrics on top of the core functions. Moreover, the toolbox supports computation on CPUs and GPUs.  Finally, `HOI` provides tools for visualizing and presenting results to simplify the interpretation and analysis of the outputs.


# Statement of need

Recent research studying higher-order interactions with information theoretic measures provides new angles and valuable insights in different fields, such as neuroscience [@gatica:2021; @herzog:2022; @combrisson:2024; @luppi:2022; @baudot:2019], music [@rosas:2019], economics [@scagliarini:2023] and psychology [@marinazzo:2022]. Information theory allows investigating higher-order interactions using a rich set of metrics that provide interpretable values of the statistical interdependency among multivariate data [@williams:2010; @mediano:2021; @barrett:2015; @rosas:2019; @scagliarini:2023; @williams:2010].

Despite the relevance of studying higher-order interactions across various fields, there is currently no toolkit that compiles the latest approaches and offers user-friendly functions for calculating higher-order information metrics. Computing higher-order information presents two main challenges. First, these metrics rely on entropy and mutual information, whose estimation must be adapted to different types of data [@madukaife:2024; @czyz:2024]. Second, the computational complexity increases exponentially as the number of variables and interaction orders grows. For example, a dataset with 100 variables, has approximately 1.6e5 possible triplets, 4e6 quadruplets, and 7e7 quintuplets. Therefore, an efficient implementation, scalable on modern hardware is required.

Several toolboxes have implemented a few HOI metrics like [`infotopo`](https://github.com/pierrebaudot/INFOTOPO) [@baudot:2019], [`infotheory`](http://mcandadai.com/infotheory/) [@candadai:2019] in C++, [`DIT`](https://github.com/dit/dit) [@james:2018], [`IDTxl`](https://github.com/pwollstadt/IDTxl) [@wollstadt:2018] and [`pyphi`](https://github.com/wmayner/pyphi) [@mayner:2018], in Python. However, `HOI` is the only pure Python toolbox specialized in the study of higher-order interactions offering functions to estimate with an optimal computational cost a wide range of metrics as the O-information [@rosas:2019], the topological information [@baudot:2019] and the redundancy-synergy index [@timme:2018]. Moreover, `HOI` allows to handle Gaussian, non-Gaussian, and discrete data using different state-of-the-art estimators [@madukaife:2024; @czyz:2024]. `HOI` also distinguishes itself from other toolboxes by leveraging [`Jax`](https://jax.readthedocs.io/), a library optimized for fast and efficient linear algebra operations on both CPU, GPU and TPU. Taken together, `HOI` combines efficient implementations of current methods and is adaptable enough to host future metrics, facilitating comparisons between different approaches and promoting collaboration across various disciplines. 

# Acknowledgements

We acknowledge the support from Google via the Summer of Code program via the International Neuroinformatics Coordination Facility initiative. M.N. have received funding from the French government under the “France 2030” investment plan managed by the French National Research Agency (reference : ANR-16-CONV000X / ANR-17-EURE-0029) and from Excellence Initiative of AixMarseille University - A*MIDEX (AMX-19-IET-004). A.B. and E.C were supported by the PRC project “CausaL” (ANR-18-CE28-0016) and received funding from the European Union’s Horizon 2020 Framework Programme for Research and Innovation under the Specific Grant Agreement No. 945539 (Human Brain Project SGA3). A.B. was supported by A*Midex Foundation of Aix-Marseille University project “Hinteract” (AMX-22-RE-AB-071). The “Center de Calcul Intensif of the Aix-Marseille University (CCIAM)” is acknowledged for high-performance computing resources. A.B, E.C and D.M were supported by EU’s Horizon 2020 Framework Programme for Research and Innovation under the Specific Grant Agreements No. 101147319 (EBRAINS 2.0 Project). C.F. was supported by the French National Research Agency (ANR-21-CE37-0027). We thank Giovanni Petri for fruitful suggestions and discussions.

# References
