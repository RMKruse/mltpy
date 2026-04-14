import os
import sys

sys.path.insert(0, os.path.abspath(".."))

project = "pymlt"
author = "René-Marcel Kruse"
copyright = "2024, René-Marcel Kruse"
release = "0.1.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_autodoc_typehints",
]

html_theme = "furo"

# Suppress noisy cross-reference warnings from scipy/numpy type annotations.
nitpicky = False

autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
}

# sphinx_autodoc_typehints settings
always_document_param_types = True
typehints_fully_qualified = False
