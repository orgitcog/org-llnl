import os
import sys


project = "slewpy"
copyright = "2025, Lawrence Livermore National Laboratory | LLNL-CODE-2009734"
author = "Lawrence Livermore National Laboratory"
release = "0.1.0"

extensions = [
    "sphinx_rtd_theme",
    "sphinx.ext.mathjax",
    "sphinxcontrib.bibtex",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "nbsphinx",
]

bibtex_bibfiles = ["refs.bib"]
bibtex_reference_style = "author_year"

templates_path = ["_templates"]
exclude_patterns = []

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]

nbsphinx_allow_errors = True
