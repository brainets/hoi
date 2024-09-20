# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys
import hoi
from sphinx_gallery.sorting import ExplicitOrder

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
    "numpydoc",
    "sphinxcontrib.bibtex",
    "sphinx_gallery.gen_gallery",
    "sphinx_design",
    "sphinx_contributors",
    # "sphinx_panels",
    # "sphinx_copybutton",
]
bibtex_bibfiles = ["refs.bib"]

templates_path = [
    "_templates/autosummary/class.rst",
    "_templates/autosummary/function.rst",
    "_templates/autosummary/layout.html",
    "_templates",
]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

autosummary_generate = True
autodoc_default_options = {"inherited-members": None}
# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_book_theme"

html_theme_options = {
    "show_toc_level": 1,
    "use_edit_page_button": False,
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/brainets/hoi",
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

html_css_files = [
    "style.css",
]

sphinx_gallery_conf = {
    # path to your examples scripts
    "examples_dirs": "../examples",
    "reference_url": dict(hoi=None),
    "gallery_dirs": "auto_examples",
    "backreferences_dir": "api/generated",
    "show_memory": True,
    "filename_pattern": "/plot_|sim_",
    "default_thumb_file": "_static/hoi-logo.png",
    "subsection_order": ExplicitOrder(
        [
            "../examples/tutorials",
            "../examples/it",
            "../examples/metrics",
            "../examples/statistics",
            "../examples/miscellaneous",
        ]
    ),
    "doc_module": ("hoi",),
    # "thumbnail_size": (100, 100),
}

autodoc_mock_imports = ["jax", "tqdm", "jax_tqdm"]


# The name of an image file (relative to this directory) to place at the top
# of the sidebar.
html_logo = "_static/hoi-logo.png"


# The name of an image file (relative to this directory) to use as a favicon of
# the docs.  This file should be a Windows icon file (.ico) being 16x16 or
# 32x32 pixels large.
html_favicon = "_static/favicon.ico"

html_show_sourcelink = True
html_copy_source = False
html_show_sphinx = False
