"""
Introduction to core information theoretical metrics
====================================================

This introduction guides you through the core information theoretical metrics
available. These metrics are the entropy and the mutual information.
"""
import numpy as np
from hoi.core import get_entropy
from hoi.core import get_mi

###############################################################################
# Entropy
# -------
#
# The fundamental information theoretical metric is the entropy. Most of the
# other higher-order metrics of information theory defined in HOI are based on
# the entropy.
#
# In HOI there are 4 different methods to compute the entropy, in this tutorial
# we will use the estimation based on the Gaussian Copula estimation.
#
# Let's start by extracting a sample `X` from a multivariate Gaussian
# distribution with zero mean and unit variance:

D = 3
x = np.random.normal(size=(D, 1000))

###############################################################################
# Now we can compute the entropy of `X`. We use the function `get_entropy` to
# build a callable function to compute the entropy. The function `get_entropy`
# takes as input the method to use to compute the entropy. In this case we use
# the Gaussian Copula estimation, so we set the method to `"gc"`:

entropy = get_entropy(method="gc")

###############################################################################
# Now we can compute the entropy of `X` by calling the function `entropy`. This
# function takes as input an array of data of shape `(n_features, n_samples)`.
# For the Gaussian Copula estimation, the entropy is computed in bits. We have:

print("Entropy of x: %.2f" % entropy(x))

###############################################################################
# For comparison, we can compute the entropy of a multivariate Gaussian with
# the analytical formula, which is:
#
# .. math::
#   H(X) = \frac{1}{2} \log \left( (2 \pi e)^D \det(\Sigma) \right)  / log(2)
#
# where :math:`D` is the dimensionality of the Gaussian and :math:`\Sigma` is
# the covariance matrix of the Gaussian. We have:

C = np.cov(x, rowvar=True)
entropy_analytical = (
    0.5 * (np.log(np.linalg.det(C)) + D * (1 + np.log(2 * np.pi)))
) / np.log(2)
print("Analytical entropy of x: %.2f" % entropy_analytical)

###############################################################################
# We see that the two values are very close.
#
# Mutual information
# ------------------
#
# The mutual information is another fundamental information theoretical metric.
# In this tutorial we will compute the mutual information between two variables
# `X` and `Y`. `X` is a multivariate Gaussian with zero mean and unit variance,
# while `Y` is a multivariate uniform distribution in the interval
# :math:`[0,1]`. Since the two variables are independent, the mutual
# information between them is expected to be zero.

D = 3
x = np.random.normal(size=(D, 1000))
y = np.random.rand(D, 1000)

mi = get_mi(method="gc")
print("Mutual information between x and y: %.2f" % mi(x, y))
