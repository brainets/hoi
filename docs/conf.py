# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys
import hoi

sys.path.insert(0, os.path.abspath(".."))
# sys.path.append(os.path.abspath("sphinxext"))

project = "HOI"
copyright = "BraiNets"
author = "BraiNets"

version = hoi.__version__
release = hoi.__version__

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration


# Add "sphinx.ext.linkcode" when you specify a linkcode_resolve function that returns an URL based on the object.
extensions = [
    "sphinx.ext.todo",
    "sphinx.ext.viewcode",
    "sphinx.ext.autodoc",
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    "sphinx.ext.coverage",
    "sphinx.ext.mathjax",
    "sphinx.ext.githubpages",
    "sphinx.ext.autosummary",
    "sphinx_panels",
    "numpydoc",
    "sphinx_copybutton",
]


templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]


autodoc_mock_imports = ["jax", "tqdm"]

