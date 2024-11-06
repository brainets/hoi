#!/usr/bin/env python
# -*- coding: utf-8 -*-
# License: 3-clause BSD
import os
from setuptools import setup, find_packages

__version__ = "0.0.5"
NAME = "hoi"
AUTHOR = "BraiNets"
MAINTAINER = "Etienne Combrisson"
EMAIL = "e.combrisson@gmail.com"
KEYWORDS = "information-theory statistics higher-order-interactions"
DESCRIPTION = "Higher Order Interactions"
URL = "https://github.com/brainets/hoi"
DOWNLOAD_URL = (
    "https://github.com/brainets/hoi/archive/v" + __version__ + ".tar.gz"
)
# Data path :
PACKAGE_DATA = {}


def read(fname):
    """Read README and LICENSE."""
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


with open("requirements.txt") as f:
    requirements = f.read().splitlines()

test_deps = ["pytest", "pytest-cov", "codecov", "xarray"]
doc_deps = [
    "sphinx!=4.1.0",
    "sphinx-gallery",
    "sphinx_design",
    "sphinx-book-theme",
    "sphinxcontrib-bibtex",
    "sphinx_contributors",
    "numpydoc",
    "matplotlib",
    "xgi",
]
lint_deps = ["flake8", "pep8-naming", "black"]

setup(
    name=NAME,
    version=__version__,
    packages=find_packages(),
    package_dir={"hoi": "hoi"},
    package_data=PACKAGE_DATA,
    include_package_data=True,
    description=DESCRIPTION,
    long_description=read("README.rst"),
    platforms="any",
    setup_requires=["numpy"],
    install_requires=requirements,
    extras_require={
        "test": requirements + test_deps,
        "doc": requirements + doc_deps,
        "lint": requirements + lint_deps,
        "full": requirements + test_deps + doc_deps + lint_deps,
    },
    dependency_links=[],
    author=AUTHOR,
    maintainer=MAINTAINER,
    author_email=EMAIL,
    url=URL,
    download_url=DOWNLOAD_URL,
    license="BSD 3-Clause License",
    keywords=KEYWORDS,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Education",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Visualization",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
