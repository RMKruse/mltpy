"""Sphinx configuration for the pymlt documentation site."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(".."))

# ---------------------------------------------------------------------------
# Project metadata
# ---------------------------------------------------------------------------
project = "pymlt"
author = "René-Marcel Kruse"
copyright = "2026, René-Marcel Kruse"
release = "0.3.0"

# ---------------------------------------------------------------------------
# Extensions
# ---------------------------------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx_autodoc_typehints",
    "nbsphinx",
    "myst_parser",
]

# ---------------------------------------------------------------------------
# Source
# ---------------------------------------------------------------------------
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}
exclude_patterns = ["_build", "**.ipynb_checkpoints", "Thumbs.db", ".DS_Store"]

# ---------------------------------------------------------------------------
# HTML / theme
# ---------------------------------------------------------------------------
html_theme = "pydata_sphinx_theme"
html_title = "pymlt"
html_theme_options = {
    "github_url": "https://github.com/RMKruse/pymlt",
    "show_prev_next": False,
    "navbar_align": "left",
    "use_edit_page_button": False,
    "icon_links": [
        {
            "name": "PyPI",
            "url": "https://pypi.org/project/pymlt/",
            "icon": "fa-brands fa-python",
        },
    ],
    "header_links_before_dropdown": 5,
}
html_context = {"default_mode": "auto"}

# ---------------------------------------------------------------------------
# Autodoc / Napoleon / type hints
# ---------------------------------------------------------------------------
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
}
autodoc_typehints = "description"
napoleon_numpy_docstring = True
napoleon_google_docstring = False
always_document_param_types = True
typehints_fully_qualified = False

# ---------------------------------------------------------------------------
# nbsphinx — notebooks are executed at build time; outputs are never committed.
# ---------------------------------------------------------------------------
nbsphinx_execute = "always"
nbsphinx_timeout = 180
nbsphinx_allow_errors = False
nbsphinx_prolog = r"""
{% set docname = env.doc2path(env.docname, base=None) %}

.. note::
   This page was generated from the notebook
   `{{ docname }} <https://github.com/RMKruse/pymlt/blob/main/docs/{{ docname }}>`_.
"""

# ---------------------------------------------------------------------------
# MyST
# ---------------------------------------------------------------------------
myst_enable_extensions = ["dollarmath", "amsmath", "colon_fence"]

# ---------------------------------------------------------------------------
# Intersphinx
# ---------------------------------------------------------------------------
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "scipy": ("https://docs.scipy.org/doc/scipy", None),
    "matplotlib": ("https://matplotlib.org/stable", None),
    "pandas": ("https://pandas.pydata.org/docs", None),
}

nitpicky = False
