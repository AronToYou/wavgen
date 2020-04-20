# Configuration file for the Sphinx documentation builder.

# -- Path setup --------------------------------------------------------------

import os
import sys
import sphinx_rtd_theme

sys.path.insert(0, os.path.abspath('../..'))


# -- Project information -----------------------------------------------------

project = 'wavgen'
copyright = '2020, Aron Lloyd'
author = 'Aron Lloyd'
version = '1.4'
release = '1.4.0'

# -- General configuration ---------------------------------------------------

# Sphinx extension module names
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.todo',
    'sphinx.ext.githubpages',
    'sphinx.ext.napoleon',
    'sphinx_rtd_theme',
]

# Paths that contain templates, relative to this directory.
templates_path = ['_templates']

# Endings to the build's source files
source_suffix = '.rst'

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# Pygments style to use
pygments_style = 'sphinx'

# Master toctree document
master_doc = 'index'

# Sphinx automated documenting extension options
autodoc_default_options = {
    'members': None,
    'private-members': None,
    'special-members': '__init__',
    'member-order': 'bysource',
}

autodoc_mock_imports = []


## HTML Options ##
html_logo = '_static/logo.png'
html_favicon = '_static/logo.ico'
# html_theme = 'sphinxjp'
# html_theme = 'insegel'
html_theme = "sphinx_rtd_theme"
html_theme_options = {
    'logo_only': True,
    'display_version': True,
    'prev_next_buttons_location': 'bottom',
    'style_external_links': False,
    'style_nav_header_background': 'white',
    # Toc options
    'collapse_navigation': True,
    'sticky_navigation': True,
    'navigation_depth': -1,
    'includehidden': True,
    'titles_only': False
}
