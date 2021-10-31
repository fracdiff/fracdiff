PROJECT_NAME := fracdiff
RUN := poetry run

.PHONY: check
check: test lint type

.PHONY: install
install:
	@poetry install --extras torch

.PHONY: test
test: test-doctest test-pytest

.PHONY: test-doctest
test-doctest:
	$(RUN) pytest --doctest-modules $(PROJECT_NAME)

.PHONY: test-pytest
test-pytest:
	$(RUN) pytest --doctest-modules --cov=$(PROJECT_NAME) tests

.PHONY: lint
lint: lint-black lint-isort

.PHONY: lint-black
lint-black:
	$(RUN) python3 -m black --check --quiet .

.PHONY: lint-isort
lint-isort:
	$(RUN) run python3 -m isort --check --force-single-line-imports --quiet .

.PHONY: format
format: format-black format-isort

.PHONY: format-black
format-black:
	$(RUN) python3 -m black --quiet .

.PHONY: format-isort
format-isort:
	$(RUN) python3 -m isort --force-single-line-imports --quiet .

.PHONY: doc
doc:
	@cd docs && make html

.PHONY: type
type:
	$(RUN) mypy $(PROJECT_NAME)

.PHONY: publish
publish:
	@gh workflow run publish.yml
