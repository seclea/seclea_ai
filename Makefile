SHELL := /bin/bash

setup_env:
	curl -sSL https://install.python-poetry.org | python3 -
	poetry install

test_:
	poetry run python3 -m unittest discover
