install:
	@poetry install

test:
	@poetry run pytest --doctest-modules fracdiff
	@poetry run pytest --doctest-modules tests

lint:
	@poetry run python3 -m black --check --quiet .
	@poetry run python3 -m isort --check --force-single-line-imports --quiet .

format:
	@poetry run python3 -m black --quiet .
	@poetry run python3 -m isort --force-single-line-imports --quiet .

publish:
	@gh workflow run publish.yml
