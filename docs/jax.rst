Jax: linear algebra backend
===========================

One of the main issues in the study of the higher-order structure of complex systems is the computational cost required to investigate one by one all the multiplets of any order. When using information theoretic tools, one must consider the fact that each metric relies on a complex set of operations that have to be performed for all the multiplets of variables in the data set. The number of possible multiplets of :math:`k` nodes in a data set grows as :math:`\binom{n}{k}`. This means that, in a data set of :math:`100` variables, the multiples of three nodes are :math:`\simeq 10^5`, the multiples of 4 nodes, :math:`\simeq 10^6` and 5 nodes, :math:`\simeq 10^7`, etc. This leads to huge computational costs and time that can pose real problems to the study of higher-order interactions in different research fields.

In this toolbox to deal with this problem, we used the recently developed Python library `Jax <https://github.com/google/jax>`_, that uses XLA to compile and run your NumPy programs on CPU, GPU and TPU.

CPU vs. GPU : Performance comparison
++++++++++++++++++++++++++++++++++++

Computing entropy on large multi-dimensional arrays
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In this first part, we are going to compare the time taken to compute entropy using large arrays. To run this comparison, we recommend using `Colab <https://colab.research.google.com/>`_ and go to *Modify > Notebook settings* and select a GPU environment.

In the first cell, install hoi and import some modules:

.. code-block:: shell

    !pip install hoi

    import numpy as np
    import jax
    import jax.numpy as jnp
    from time import time

    from hoi.metrics import Oinfo
    from hoi.core import get_entropy

    import matplotlib.pyplot as plt

    plt.style.use("ggplot")

In a new cell, past the following code. This code compute the Gaussian Copula entropy for an array with a size growing, both on the CPU or GPU :

.. code-block:: shell

    def compute_timings(n=15):
        n_samples = np.linspace(10, 10e2, n).astype(int)
        n_features = np.linspace(1, 10, n).astype(int)
        n_variables = np.linspace(1, 10e2, n).astype(int)

        entropy = jax.vmap(get_entropy(method="gc"), in_axes=(0,))

        # dry run
        entropy(np.random.rand(2, 2, 10))

        timings_cpu = []
        data_size = []
        for n_s, n_f, n_v in zip(n_samples, n_features, n_variables):
            # generate random data
            x = np.random.rand(n_v, n_f, n_s)
            x = jnp.asarray(x)

            # compute entropy
            start = time()
            entropy(x)
            timings_cpu.append(time() - start)
            data_size.append(n_s * n_f * n_v)

        return data_size, timings_cpu

    with jax.default_device(jax.devices("gpu")[0]):
        data_size, timings_gpu = compute_timings()

    with jax.default_device(jax.devices("cpu")[0]):
        data_size, timings_cpu = compute_timings()


Finally, plot the timing comparison :

.. code-block:: shell

    plt.plot(data_size, timings_cpu, label="CPU")
    plt.plot(data_size, timings_gpu, label="GPU")
    plt.xlabel("Data size")
    plt.ylabel("Time (s)")
    plt.title("CPU vs. GPU for computing entropy", fontweight="bold")
    plt.legend()


.. image:: _static/jax_cgpu_entropy.png

On CPU, the computing time increase linearly as the array gets larger. However, on GPU, it doesn't scale as fast.

Computing Higher-Order Interactions on large multiplets
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In the next example, we are going to compute Higher-Order Interactions on a large network of 10 nodes with an increasing order (i.e. multiplets up to size 3, 4, ..., 10), both on CPU and GPU.

.. code-block:: shell

    def compute_timings():
        # create a dynamic network with 1000 samples, 10 nodes and
        # 100 time points
        x = np.random.rand(1000, 10, 100)

        # define the model
        model = Oinfo(x, verbose=False)

        # compute hoi for increasing order
        order = np.arange(3, 11)
        timings = []
        for o in order:
            start = time()
            model.fit(minsize=3, maxsize=o)
            timings.append(time() - start)
        
        return order, timings

    with jax.default_device(jax.devices("gpu")[0]):
        order, timings_gpu = compute_timings()

    with jax.default_device(jax.devices("cpu")[0]):
        order, timings_cpu = compute_timings()

Let's plot the results :

.. code-block:: shell

    plt.plot(order, timings_cpu, label="CPU")
    plt.plot(order, timings_gpu, label="GPU")
    plt.xlabel("Multiplet order")
    plt.ylabel("Time (s)")
    plt.title("CPU vs. GPU for computing the O-information", fontweight="bold")
    plt.legend()


.. image:: _static/jax_cgpu_oinfo.png

On this toy example, we can see that computing the O-information is ~3x faster
on GPU than on CPU.
