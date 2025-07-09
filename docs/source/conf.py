project = 'Symbolic Topological Framework'
copyright = '2025, Noah Parsons or GuiloScion'
author = 'Noah Parsons or GuiloScion'
release = '0.1.0'

# -- General configuration ---------------------------------------------------
import os
import sys
sys.path.insert(0, os.path.abspath('../../src'))

extensions = [
    'sphinx.ext.autodoc',     # For auto-generating API docs from Python docstrings
    'sphinx.ext.napoleon',    # To support Google/NumPy style docstrings (important for clarity)
    'myst_parser',            # To allow Sphinx to parse your Markdown (.md) files
    'sphinx.ext.viewcode',    # To link to the actual source code from API docs
    'sphinx.ext.todo',        # To allow for todo notes in your docs
    # Add any other Sphinx extensions you might need later
]

templates_path = ['_templates']
exclude_patterns = []

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}