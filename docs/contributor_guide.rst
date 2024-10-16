.. _contribute:

Developer Documentation
=======================

Installation for developers
---------------------------

For developers, you can install it in develop mode with the following commands :

.. code-block:: shell

    git clone https://github.com/brainets/hoi.git
    cd hoi
    pip install -e .['full']

The full installation of HOI includes additional packages to test the software and build the documentation :

- pytest
- pytest-cov
- codecov
- pandas
- xarray
- sphinx!=4.1.0
- sphinx-gallery
- sphinx-book-theme
- sphinxcontrib-bibtex
- numpydoc
- matplotlib
- flake8
- pep8-naming
- black


Contributing to HOI
-------------------

- For questions, please use the `discussions page <https://github.com/brainets/hoi/discussions>`_
- You can signal a bug or suggests improvements to the code base or the documentation by `opening an issue <https://github.com/brainets/hoi/issues>`_

Contributing code using pull requests
-------------------------------------

We do all of our development using git, so basic knowledge is assumed.

Follow these steps to contribute code:

1. Fork the hoi repository by clicking the **Fork** button on the `repository page <https://github.com/brainets/hoi>`_
2. Install Python >= 3.8
3. Clone the hoi repository to your computer and install hoi :

.. code-block:: shell

    git clone https://github.com/YOUR_USERNAME/hoi
    cd hoi
    pip install -e .['full']

4. Add the HOI repo as an upstream remote, so you can use it to sync your changes. 

.. code-block:: shell

    git remote add upstream https://github.com/brainets/hoi

5. Create a branch where you will develop from and implement your changes using your favorite editor  :

.. code-block:: shell

    git checkout -b branch_name

6. Make sure your code passes HOI's lint and type checks, by running the following from the top of the repository:

.. code-block:: shell

    black hoi/

7. Make sure the tests pass by running the following command from the top of the repository:

.. code-block:: shell

    pytest -v

Each python file inside HOI is tested to ensure that functionalities of HOI are maintained with each commit. If you modify a file, or example `hoi/core/entropies.py`, you can run the tests for this specific file only located in `hoi/core/tests/tests_entropies.py` If you want to only test the files you modified you can use :

.. code-block:: shell

    pytest -v hoi/path_to_test_file

8. Once you are satisfied with your change, create a commit as follows :

.. code-block:: shell

    git add file1.py file2.py ...
    git commit -m "Your commit message"

Then sync your code with the main repo:

.. code-block:: shell

    git fetch upstream
    git rebase upstream/main

Finally, push your commit on your development branch and create a remote branch in your fork that you can use to create a pull request from:

.. code-block:: shell

    git push --set-upstream origin branch_name

Please ensure your contribution is a single commit

9. Create a Pull request. To this end, in your web browser go to your hoi repository and you should see a message proposing to draft a pull request.
