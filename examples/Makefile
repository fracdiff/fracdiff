.PHONY: run
run:
	find . -name "example_*.ipynb" -not -path "*/.ipynb_checkpoints/*" | papermill

.PHONY: format
format:
	black-nb .