import os
import sys

import sphinx_rtd_theme

sys.path.insert(0, os.path.abspath(".."))

# ─── NumPy C-API 初始化（可选，有时候还是需要） ───────────────────────
try:
    import numpy.core.multiarray as _multi

    _multi.import_array()
except Exception:
    pass


# ─── RTD 主题设置 ───────────────────────────────────────────────────

# -- Project information -----------------------------------------------------
project = "TSOP"
author = "Yili"
release = "1.0"

# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.mathjax",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Options for HTML output -------------------------------------------------
html_theme = "sphinx_rtd_theme"
html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]
html_static_path = ["_static"]

latex_engine = "xelatex"
latex_elements = {
    "preamble": r"""
        \usepackage{xeCJK}
        \usepackage{bbold}
        \xeCJKsetup{CJKmath=true}
        \usepackage{amsmath}
        \usepackage{amssymb}
    """,
}
