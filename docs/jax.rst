Jax: linear algebra backend
===========================

One of the main issues in the study of the higher-order structure of complex systems is the computational cost required to investigate one by one all the multiplets of any order. When using information theoretic tools, one must consider the fact that each metric relies on a complex set of operations that have to be performed for all the multiplets of variables in the data set. The number of possible multiplets of :math:`k` nodes in a data set grows as :math:`\binom{n}{k}`. This means that, in a data set of :math:`100` variables, the multiples of three nodes are :math:`\simeq 10^5`, the multiples of 4 nodes, :math:`\simeq 10^6` and 5 nodes, :math:`\simeq 10^7`, etc. This leads to huge computational costs and time that can pose real problems to the study of higher-order interactions in different research fields.

In this toolbox to deal with this problem, we used the recently developed Python library `Jax <https://github.com/google/jax>`_, that uses XLA to compile and run your NumPy programs on CPU, GPU and TPU.
