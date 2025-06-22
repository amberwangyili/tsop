
PYTHON        ?= python3
FLAKE8        = $(PYTHON) -m flake8
CPPLINT       = cpplint
CPPLINTFLAGS  = --quiet --filter=-whitespace/line_length,-legal/copyright,-build/header_guard

.PHONY: help install build wheel test fmt lint docs clean

help:
	@echo "Usage:"
	@echo "  make install   # install packages into current environment"
	@echo "  make build     # build extensions inplace"
	@echo "  make wheel     # build wheel package"
	@echo "  make test      # run pytest"
	@echo "  make fmt      # run formatter"
	@echo "  make lint      # run flake8"
	@echo "  make docs      # build Sphinx HTML docs"
	@echo "  make clean     # remove build artifacts"


build:
	@$(PYTHON) setup.py build_ext --inplace

wheel: clean
	@$(PYTHON) -m pip install --upgrade build
	@$(PYTHON) -m build --wheel

test:
	@pytest

lint:
	@echo ">>> Lint: Python (.py) + Cython (.pyx)"
	@$(FLAKE8) --filename=*.py,*.pyx --ignore=E999 . --exclude tests
	@echo ">>> Lint: C/C++ (.h)"
	@find tsop -type f -name '*.h' -print \
		| xargs $(CPPLINT) $(CPPLINTFLAGS)


docs:
	@cd docs && $(MAKE) html

clean:
	# remove Python bytecode caches
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.py[co]"    -delete

	# remove pytest cache
	rm -rf .pytest_cache

	# remove Cython-generated C files and compiled extensions
	find . -type f -name "*.c"   -delete
	find . -type f -name "*.so"  -delete
	find . -type f -name "*.pyd" -delete

	# remove build artifacts
	rm -rf build/ dist/ *.egg-info/ docs/_build

install:
	@echo ">>> Installing tsop into your current environment"
	@$(PYTHON) -m pip install .



fmt:
	@echo ">>> Auto-format Python with Black (skip .pyx)"
	@black .
	@echo ">>> Sort imports with isort"
	@isort .
	@echo ">>> Autopep8 aggressive fix"
	@autopep8 --in-place --aggressive --recursive \
		--exclude=tsop/basic.cpp,tsop/basic.h \
		.
	@echo ">>> Running clang-format on .h files"
	# 对所有 .h 原地格式化
	@find tsop -type f \( -name '*.h' \) \
		-exec clang-format -i {} \;
