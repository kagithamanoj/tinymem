.PHONY: test eval demo clean

VENV ?= .venv
PY = $(VENV)/bin/python

test:
	$(PY) -m pytest tests/ -q

eval:
	$(PY) evals/benchmark.py

demo:
	$(PY) evals/demo.py

clean:
	rm -rf __pycache__ tinymem/__pycache__ tests/__pycache__ evals/__pycache__ .pytest_cache
