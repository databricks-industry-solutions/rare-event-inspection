# Contributing to Kakapo
![kakapo-logo](./resources/kakapo-logo.png)

## Overview
We happily welcome contributions to Kakapo as long as CLA has been accepted.
We use GitHub Issues to track community reported issues and GitHub Pull Requests for accepting changes.

## Repository structure
The repository is structured as follows:

- `.github` CICD for both kakapo and for solution examples.
- `python/kakapo/` Kakapo python module and tests. This code is used to build databricks-kakapo pip dependency.
- `docs/` Source code for documentation. Documentation is built via sphinx.

## Test & build Kakapo

### Python

The python bindings can be tested using [unittest](https://docs.python.org/3/library/unittest.html).
- Move to the `python/` directory and install the project and its dependencies:
  `pip install . `
- Run the tests using `unittest`: `python -m unittest`

The project wheel file can be built with [build](https://pypa-build.readthedocs.io/en/stable/).
- Install the build requirements: `pip install build wheel`.
- Build the wheel using `python -m build`.
- Collect the .whl file from `python/dist/`

### Documentation

The documentation has been produced using [Sphinx](https://www.sphinx-doc.org/en/master/).

To build the docs:
- Install the pandoc library (follow the instructions for your platform [here](https://pandoc.org/installing.html)).
- Install the python requirements from `docs/docs-requirements.txt`.
- Build the HTML documentation by running `make html` from `docs/`.
- You can locally host the docs by running the `reload.py` script in the `docs/source/` directory.

## Style

Tools we use for code formatting and checking:
- `scalafmt` and `scalastyle` in the main scala project.
- `black` and `isort` for the python bindings.
