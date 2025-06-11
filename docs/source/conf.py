# PRIVATEER Documentation Configuration
import os
import sys
sys.path.insert(0, os.path.abspath('../../'))

# Project information
project = 'PRIVATEER Anomaly Detection'
copyright = '2024, INFILI - EU Horizon Europe'
author = 'PRIVATEER Consortium'
release = '1.0.0'

# Extensions
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
    'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax',
]

# Napoleon settings for docstrings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False

# Autodoc settings
autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'special-members': '__init__',
    'undoc-members': True,
    'exclude-members': '__weakref__',
    'show-inheritance': True
}

autosummary_generate = True
autosummary_imported_members = True

# Keep module structure in docs
toctree_collapse = False

# Theme
html_theme = 'sphinx_rtd_theme'
html_theme_options = {
    'collapse_navigation': False,
    'sticky_navigation': True,
    'navigation_depth': 4,
}

# Output
html_show_sourcelink = True