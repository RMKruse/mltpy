Installation
============

mltpy is a pure-Python package with only numpy and scipy as required
dependencies. It targets Python ≥ 3.12.

Basic install
-------------

.. code-block:: bash

   pip install mltpy

Optional extras
---------------

.. code-block:: bash

   pip install "mltpy[plots]"      # matplotlib for .plot() helpers
   pip install "mltpy[pandas]"     # accept pd.Series as y
   pip install "mltpy[examples]"   # run the vignettes (lifelines, jupyter, …)
   pip install "mltpy[dev]"        # tests, linters, type checker
   pip install "mltpy[docs]"       # build the documentation locally

Development install
-------------------

.. code-block:: bash

   git clone https://github.com/RMKruse/mltpy
   cd mltpy
   pip install -e ".[dev,examples]"

Building the docs locally also requires `pandoc <https://pandoc.org/>`_,
which nbsphinx uses to execute and render the Jupyter vignettes:

.. code-block:: bash

   pip install -e ".[docs]"
   sphinx-build -b html -W --keep-going docs/ docs/_build/html
