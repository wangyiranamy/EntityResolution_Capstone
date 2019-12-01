# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import sphinx_rtd_theme

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
sys.path.insert(0, os.path.abspath('../src/entity_resolver'))


# -- Project information -----------------------------------------------------

project = 'entity-resolver'
copyright = '2019, Yiran Wang'
author = 'Yiran Wang'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.doctest',
    'sphinx.ext.linkcode',
    'sphinx.ext.todo',
    'sphinx_rtd_theme',
    'sphinxcontrib.napoleon',
    'sphinx_autodoc_typehints',
    'sphinx.ext.intersphinx',
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

html_theme_options = {
    'collapse_navigation': False,
    'navigation_depth': 4
}

html_context = {
    "display_github": True,
    "github_user": "wangyiranamy",
    "github_repo": "EntityResolution_Capstone",
    "github_version": "master",
    "conf_py_path": "/doc/source/",
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']


# -- Other options -----------------------------------------------------------

def linkcode_resolve(domain, info):
    if domain != 'py':
        return None
    if not info['module']:
        return None
    filename = info['module'].replace('.', '/')
    url = (
        'https://github.com/wangyiranamy/EntityResolution_Capstone/'
        'blob/master/src/'
    )
    return f'{url}{filename}.py'


default_role = 'py:obj'

napoleon_include_special_with_doc = True

napoleon_include_private_with_doc = True

intersphinx_mapping = {'python': ('https://docs.python.org/3.6', None)}

todo_include_todos = True
