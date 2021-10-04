PROJECT_NAME := fracdiff

.PHONY: check
check: test lint typecheck

.PHONY: install
install:
	@poetry install --extras torch

.PHONY: test
test: test-doctest test-pytest

.PHONY: test-doctest
test-doctest:
	@poetry run pytest --doctest-modules $(PROJECT_NAME)

.PHONY: test-pytest
test-pytest:
	@poetry run pytest --doctest-modules --cov=$(PROJECT_NAME) tests

.PHONY: lint
lint: lint-black lint-isort

.PHONY: lint-black
lint-black:
	@poetry run python3 -m black --check .

.PHONY: lint-isort
lint-isort:
	@poetry run python3 -m isort --check --force-single-line-imports .

.PHONY: format
format: format-black format-isort

.PHONY: format-black
format-black:
	@poetry run python3 -m black --quiet .

.PHONY: format-isort
format-isort:
	@poetry run python3 -m isort --force-single-line-imports --quiet .

.PHONY: doc
doc:
	@cd docs && make html

.PHONY: typecheck
typecheck:
	@poetry run mypy $(PROJECT_NAME)

.PHONY: publish
publish:
	@gh workflow run publish.yml
