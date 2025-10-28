.PHONY: setup run-app fmt lint

setup:
	python -m venv .venv
	. .venv/bin/activate && python -m pip install --upgrade pip
	. .venv/bin/activate && python -m pip install -r requirements.txt

run-app:
	shiny run --reload app/app.py

fmt:
	python -m pip install --upgrade ruff black
	ruff check --fix src app
	black src app

lint:
	ruff check src app

