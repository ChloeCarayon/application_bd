
PY_SRC := src/

.PHONY: install
install: # Install dependencies
	poetry install

.PHONY: run
run: # Run code
	poetry run python  src/main.py

.PHONY: build
build:  ## Build the package wheel and sdist.
	poetry build

.PHONY: check-pylint
check-pylint:  ## Check for code smells using pylint.
	poetry run pylint $(PY_SRC)

.PHONY: check
check:
	poetry run isort $(PY_SRC)
	poetry run black $(PY_SRC)
	poetry run flake8 $(PY_SRC)

.PHONY: publish
publish:  ## Publish the latest built version on PyPI.
	poetry publish
