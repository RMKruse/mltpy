.PHONY: benchmark test validate docs

benchmark:
	python benchmarks/bench_python.py
	Rscript benchmarks/bench_r.R
	python benchmarks/report.py

test:
	pytest tests/ -v

validate:
	python validation/run_validation.py

docs:
	sphinx-build -b html -W --keep-going docs/ docs/_build/html
