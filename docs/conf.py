# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys
import hoi

# sys.path.append(os.path.abspath("sphinxext"))

sys.path.insert(0, os.path.abspath(".."))

project = "HOI"
copyright = "BraiNets"
author = "BraiNets"
release = hoi.__version__
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
    "sphinxcontrib.bibtex",
]
bibtex_bibfiles = ["refs.bib"]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"
# html_theme = "sphinx_rtd_theme"
html_theme_options = {
    "show_toc_level": 1,
    "use_edit_page_button": False,
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/brainets/frites",
            "icon": "fab fa-github-square",
        },
        {
            "name": "Twitter",
            "url": "https://twitter.com/kNearNeighbors",
            "icon": "fab fa-twitter-square",
        },
    ],
}
html_static_path = ["_static"]


autodoc_mock_imports = ["jax", "tqdm", "jax_tqdm"]

