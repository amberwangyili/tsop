# ─── docs/Makefile for TSOP ────────────────────────────────────────────────

# Python 可执行命令
PYTHON        ?= python3
# 项目根目录（包含 setup.py）
PROJECT_ROOT  := $(abspath $(CURDIR)/..)

# Sphinx 配置
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     := .
BUILDDIR      := _build
SPHINXOPTS    ?= -j auto

.PHONY: help html latex latexpdf clean

help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS)

# ─── HTML ───────────────────────────────────────────────────────────────────
html:
	@echo ">>> 编译 Cython 扩展到原地"
	@cd "$(PROJECT_ROOT)" && $(PYTHON) setup.py build_ext --inplace
	@echo ">>> 使用 Sphinx 生成 HTML"
	@$(SPHINXBUILD) -M html "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS)
	@echo ">>> HTML 输出目录：$(BUILDDIR)/html"

# ─── LaTeX 源 (.tex) ────────────────────────────────────────────────────────
latex:
	@echo ">>> 编译 Cython 扩展到原地"
	@cd "$(PROJECT_ROOT)" && $(PYTHON) setup.py build_ext --inplace
	@echo ">>> 使用 Sphinx 生成 LaTeX 源"
	@$(SPHINXBUILD) -M latex "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS)
	@echo ">>> LaTeX 源输出目录：$(BUILDDIR)/latex"

# ─── PDF via XeLaTeX ────────────────────────────────────────────────────────
latexpdf: latex
	@echo ">>> 使用 XeLaTeX 编译 PDF"
	@cd "$(BUILDDIR)/latex" && \
		xelatex -interaction=nonstopmode tsop.tex > /dev/null && \
		xelatex -interaction=nonstopmode tsop.tex > /dev/null
	@echo ">>> PDF 输出路径：$(BUILDDIR)/latex/tsop.pdf"

# ─── 清理所有构建产物 ─────────────────────────────────────────────────────────
clean:
	@echo ">>> 清理 Sphinx 构建目录和 Cython 产物"
	@rm -rf "$(BUILDDIR)"
	@cd "$(PROJECT_ROOT)" && $(PYTHON) setup.py clean --all
	@find "$(PROJECT_ROOT)" -maxdepth 2 -type f \( -name '*.so' -o -name '*.pyd' \) -delete
