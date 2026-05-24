# Contributing to pymlt

## Prerequisites

- Python >= 3.12
- git

Optional (only needed to regenerate R reference files):

- R with the `mlt` and `basefun` packages installed

## Getting started

```bash
git clone https://github.com/rene-marcel-kruse/pymlt
cd pymlt
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

## Running tests

```bash
pytest tests/          # full suite
pytest --cov=pymlt     # with coverage report
```

## Benchmarks

```bash
make benchmark         # times pymlt vs. R mlt across n × order × censoring
```

Requires R with the `mlt`, `basefun`, `variables`, and `survival` packages.
See [`benchmarks/README.md`](benchmarks/README.md) for layout and CSV schema;
the latest committed report is at
[`benchmarks/results/benchmark_report.md`](benchmarks/results/benchmark_report.md).

## Code style

| Tool | Command | Notes |
|------|---------|-------|
| ruff | `ruff check .` / `ruff format .` | linting + formatting |
| mypy | `mypy pymlt/` | strict mode; scipy stubs excluded |

Line length: 88 characters.

## Module dependency order

The six modules form a strict dependency chain. If you change a lower layer,
re-run all tests above it before submitting:

```
variables → basis → constraints → likelihood → optimizer → model
```

## Adding tests

- One `tests/test_<module>.py` file per module — mirror the existing pattern.
- Mathematical invariants (monotonicity, CDF range, …) belong in
  Hypothesis `@given` tests.
- Numerical comparisons against R belong in the skip-guarded integration test
  in `tests/test_model.py::test_integration_r_reference`.

## R reference files

If you change `likelihood.py` or `basis.py`, regenerate the reference values
before running the integration test:

```bash
Rscript reference/generate_reference.R
pytest tests/test_model.py -k r_reference
```

## Submitting a pull request

- One logical change per PR.
- All tests must pass: `pytest tests/ -q`
- No mypy errors: `mypy pymlt/`
- No ruff warnings: `ruff check .`
