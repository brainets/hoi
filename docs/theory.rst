Theoretical background
======================

Since the seminal work of Claude Shannon :cite:`shannon1948mathematical`, in the second half of the 20th century, :term:`Information Theory` (IT) has been proven to be an invaluable framework to decipher the intricate web of interactions underlying a broad range of different complex systems :cite:`watanabe1960information`. In this line of research, a plethora of approaches has been developed to investigate relationships between pairs of variables, shedding light on many properties of complex systems. However, a growing body of literature has recently highlighted that investigating the interactions between groups of more than 2 units, i.e. :term:`Higher Order Interactions` (HOI), allows to unveil effects that can be neglected by pairwise approaches :cite:`battiston2020networks`. Hence, how to study HOI has become a more and more important question in recent times :cite:`battiston2021physics`. In this context, new approaches based on IT emerged to investigate HOI in terms of information content; more into details, different metrics have been developed to estimate from the activity patterns of a set of variables, whether or not they were interacting and which kind of interaction they presented  :cite:`timme2014synergy, varley2023information`. Most of these metrics are based on the concepts of :term:`Synergy` and :term:`Redundancy`, formalized in terms of IT by the :term:`Partial Information Decomposition` (PID) framework :cite:`williams2010nonnegative`. Even though these metrics are theoretically well defined and fascinating, when concretely using them to study and computing the higher-order structure of a system, two main problems come into play: how to estimate entropies and information from limited data set, with different hypothesis and characteristics, and how to handle the computational cost of such operations. In our work we provided a complete set of estimators to deal with different kinds of data sets, e.g. continuous or discrete, and we used the python library `Jax <https://github.com/google/jax>`_ to deal with the high computational cost. In the following part we will introduce the information theoretical tools necessary to understand and use the metrics developed in this toolbox. Then we will present quickly the theory behind the metrics developed, their use and possible interpretation. Finally, we are going to discuss limitations and future developments of our work.

Core information theoretic measures
+++++++++++++++++++++++++++++++++++

In this section, we delve into some fundamental information theoretic measures, such as Shannon :term:`Entropy` and :term:`Mutual Information` (MI), and their applications in the study of pairwise interactions. Besides the fact that these measures play a crucial role in various fields such as data science and machine learning, as we will see in the following parts, they serve as the building blocks for quantifying information and interactions between variables at higher-orders.

Measuring Entropy
*****************

Shannon entropy is a fundamental concept in IT, representing the amount of uncertainty or disorder in a random variable :cite:`shannon1948mathematical`. Its standard definition for a discrete random variable :math:`X`, with probability mass function :math:`P(X)`, is given by:

.. math::

	H(X) = −\sum P(x) log_{2}(P(x))

However, estimating the probability distribution :math:`P(X)` from data can be challenging. When dealing with a discrete variable that takes values from a limited set :math:`{x_{1}, x_{2}, ...}`, one can estimate the probability distribution by computing the frequencies of each state :math:`x_{i}`. In this scenario we estimate the probability :math:`P(X=x_{i}) = n_{i}/N`, where :math:`n_{i}` is the number of occurrences :math:`X=x_{i}` and :math:`N` is the number of data points. This can present problems due to size effects when using a small data set and variables exploring a big set of states.

A more complicated and common scenario is the one of continuous variables. To estimate the entropy of a continuous variable, different methods are implemented in the toolbox:

- Binning method, that consists in binning the continuous data in a discrete set of bins. In this way, variables are discretized and the entropy can be computed as described above. This procedure can be performed in many different ways :cite:`endres2005bayesian, darbellay1999estimation, fraser1986independent`.
- K-Nearest Neighbors (KNN), that estimates the probability distribution by considering the K nearest neighbors of each data point :cite:`kraskov2004estimating`.
- Kernel Density Estimation that uses kernel functions to estimate the probability density function, offering a smooth approximation :cite:`moon1995estimation`.
- The parametric estimation, that is used when the data is gaussian and allows to compute the entropy as a function of the variance :cite:`goodman1963statistical`.

Note that all the functions mentioned in the following part are based on the computation of  entropies, hence we advise care in the choice of the estimator to use.

Measuring Mutual Information (MI)
*********************************

One of the most used functions in the study of pairwise interaction is the Mutual Information (MI) that quantifies the statistical dependence or information shared between two random variables :cite:`shannon1948mathematical, watanabe1960information`. It is defined mathematically using the concept of entropies. For two random variables X and Y, MI is given by:

.. math::

	MI(X;Y) = H(X) + H(Y) − H(X,Y)

Where:

:math:`H(X)` and :math:`H(Y)` are the entropies of individual variables :math:`X` and :math:`Y`.
:math:`H(X,Y)`  is the joint entropy of :math:`X` and :math:`Y`.
MI between two variables, quantifies how much knowing one variable reduces the uncertainty about the other and measures the interdependency between the two variables. If they are independent, we have :math:`H(X,Y)=H(X)+H(Y)`, hence :math:`MI(X,Y)=0`. Since the MI can be reduced to a signed sum of entropies, the problem of how to estimate MI from continuous data can be reconducted to the problem, discussed above, of how to estimate entropies. An estimator that has been recently developed and presents interesting properties when computing the MI is the Gaussian Copula estimator :cite:`ince2017statistical`. This estimator is based on the statistical theory of copulas and is proven to provide a lower bound to the real value of MI, this is one of its main advantages: when computing MI, Gaussian copula estimator avoids false positives. Play attention to the fact that this can be mainly used to investigate relationships between two variables that are monotonic.

From pairwise to higher-order interactions 
++++++++++++++++++++++++++++++++++++++++++	

The information theoretic metrics involved in this work are all based in principle on the concept of Shannon entropy and mutual information. Given a set of variables, a common approach to investigate their interaction is by comparing the entropy and the information of the joint probability distribution of the whole set with the entropy and information of different subsets. This can be done in many different ways, unveiling different aspects of HOI :cite:`timme2014synergy, varley2023information`. The metrics implemented in the toolbox can be divided in two main categories: a group of metrics measures the interaction behavior prevailing within a set of variable, :term:`Network behavior`, another group of metrics instead focuses on the relationship between a set of source variables and a target one, :term:`Network encoding`. In the following parts we are going through all the metrics that have been developed in the toolbox, providing some insights about their theoretical foundation and possible interpretations.

Network behavior 
*****************

Total correlation 
-----------------

Total correlation, :class:`hoi.metrics.TC`, is the oldest exstension of mutual information to
an arbitrary number of variables :cite:`watanabe1960information, studeny1998multiinformation`. It is defined as:

.. math::

	TC(X^{n})  &=  \sum_{j=1}^{n} H(X_{j}) - H(X^{n}) \\

The total correlation quantifies the strength of collective constraints ruling the systems, it is sentive to information shared between single variables and it can be associated with redundancy.


Dual Total correlation
----------------------

Dual total correlation, :class:`hoi.metrics.DTC`, is another extension of mutual information to
an arbitrary number of variables, also known as binding information and excess entropy, :cite:`sun1975linear`. It quatifies the part of the joint entropy that is shared by at least two or more variables in the following way:

.. math::

	DTC(X^{n})  &=  H(X^{n}) - \sum_{j=1}^{n} H(X_j|X_{-j}^{n}) \\
				&= \sum_{j=1}^{n} H(X_j) - (n-1)H(X^{n})

where :math:`\sum_{j=1}^{n} H(X_j|X_{-j}^{n})` is the entropy of :math:`X_j` not shared by any other variable. This measure is higher in systems in which lower order constraints prevails.

S information
-------------

The S-information (also called exogenous information), :class:`hoi.metrics.Sinfo`, is defined
as the sum between the total correlation (TC) plus the dual total
correlation (DTC), :cite:`james2011anatomy`:

.. math::

	\Omega(X^{n})  &=  TC(X^{n}) + DTC(X^{n}) \\
					&=  nH(X^{n}) + \sum_{j=1}^{n} [H(X_{j}) + H(
					X_{-j}^{n})]

It is sensitive to both redundancy and synergy, quantifying the total ammount of constraints ruling the system under study.

O-information
-------------

One prominent metric that has emerged in the pursuit of higher-order understanding is the O-information, :class:`hoi.metrics.Oinfo`. Introduced by Rosas in 2019 :cite:`rosas2019oinfo`, O-information elegantly addresses the challenge of quantifying higher-order dependencies by extending the concept of mutual information. Given a multiplet of :math:`n` variables, :math:`X^n = \{ X_0, X_1, …, X_n \}`, its formal definition is the following:  

.. math::

	\Omega(X^n)= (n-2)H(X^n)+\sum_{i=1}^n \left[ H(X_i) - H(X_{-i}^n) \right]
    
Where :math:`X_{-i}` is the set of all the variables in :math:`X^n` apart from :math:`X_i`. The O-information can be written also as the difference between the total correlation and the dual total correlation and reflects the balance between higher-order and lower-order constraints among the set of variables of interest. It is shown to be a proxy of the difference between redundancy and synergy: when the O-information of a set of variables is positive this indicates redundancy, when it is negative, synergy. In particular when working with big data sets it can become complicated 

Topological information
-----------------------

The topological information, :class:`hoi.metrics.InfoTopo`, a generalization of the mutual information to higher-order, :math:`I_k` has been introduced and presented to test uniformity and dependence in the data :cite:`baudot2019infotopo`. Its formal definition is the following:

.. math::

    I_{k}(X_{1}; ...; X_{k}) = \sum_{i=1}^{k} (-1)^{i - 1} \sum_{I\subset[k];card(I)=i} H_{i}(X_{I})

Note that :math:`I_2(X,Y) = MI(X,Y)` and that :math:`I_3(X,Y,Z)=\Omega(X,Y,Z)`. As the O-information this function can be interpreted in terms of redundancy and synergy, more into details when it is positive it indicates that the system is dominated by redundancy, when it is negative, synergy.

Network encoding 
****************

Gradient of O-information
-------------------------

The O-information gradient, :class:`hoi.metrics.GradientOinfo`, has been developed to study the contribution of one or a set of variables to the O-information of the whole system :cite:`scagliarini2023gradients`. In this work we proposed to use this metric to investigate the relationship between multiplets of source variables and a target variable. Following the definition of the O-information gradient of order 1 we have:

.. math::

    \partial_{target}\Omega(X^n) = \Omega(X^n, target) - \Omega(X^n)

This metric does not focus on the O-information of a group of variables, instead it reflects the variation of O-information when the target variable is added to the group. This allows to unveil the contribution of the target to the group of variables in terms of O-information, providing insights about the relationship between the target and the group of variables. Note that, when the target is statistically  independent from all the variables of the group, the gradient of O-information is 0, when it is greater than 0, the relation between variables and target is characterized by redundancy, when negative, synergy.

Redundancy-Synergy index (RSI)
------------------------------

Another metric, proposed by Gal Chichek et al in 2001 :cite:`chechik2001group`, is the Redundancy-Synergy index, :class:`hoi.metrics.RSI`, developed as an extension of mutual information, aiming to characterize the statistical interdependencies between a group of variables :math:`X^n` and a target variable :math:`Y`, in terms of redundancy and synergy, it is computed as:

.. math::

	RSI(X^n, Y) = I(X^n, Y) - \sum_{i=0}^n I(X_i,Y)

The RSI is designed to measure directly whether the sum of the information provided separately by all the variables is greater or not with respect to the information provided by the whole group. When RSI is positive, the whole group is more informative than the sum of its parts separately, so the interaction between the variables and the target is dominated by synergy. A negative RSI should instead suggest redundancy among the variables with respect to the target.

Synergy and redundancy (MMI)
----------------------------

Within the broad research field of IT a growing body of literature has been produced in the last 20 years about the fascinating concepts of synergy and redundancy. These concepts are well defined in the framework of Partial Information Decomposition, which aims to distinguish different “types” of information that a set of sources convey about a target variable. In this framework, the synergy between a set of variables refers to the presence of relationships between the target and the whole group that cannot be seen when considering separately the single parts. Redundancy instead refers to another phenomena, in which variables contain copies of the same information about the target. Different definition have been provided in the last years about these two concepts, in our work we are going to report the simple case of the Minimum Mutual Information (MMI) :cite:`barrett2015exploration`, in which the redundancy, :class:`hoi.metrics.RedundancyMMI`, between a set of :math:`n` variables :math:`X^n = \{ X_1, \ldots, X_n\}` and a target :math:`Y` is defined as: 

.. math::

	redundancy (Y, X^n) = min_{i<n} I \left( Y, X_i \right)
    
When computing the redundancy in this way the definition of synergy, :class:`hoi.metrics.SynergyMMI`, follows:

.. math::

	synergy (Y, X^n) =  I \left( Y, X^n \right) - max_{i<n} I \left( Y, X^n_{ -i } \right)

Where :math:`X^n_{-i}` is the set of variables :math:`X^n`, excluding the variable :math:`i`. This metric has been proven to be accurate when working with gaussian systems; we advise care when interpreting the results of the redundant interactions, since the definition of redundancy reflects simply the minimum information provided by the source variables.

Bibliography
============

.. bibliography:: refs.bib
    :style: plain
