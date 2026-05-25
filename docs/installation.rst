Installation
============

pymlt is a pure-Python package with only numpy and scipy as required
dependencies. It targets Python ≥ 3.12.

Basic install
-------------

.. code-block:: bash

   pip install pymlt

Optional extras
---------------

.. code-block:: bash

   pip install "pymlt[plots]"      # matplotlib for .plot() helpers
   pip install "pymlt[pandas]"     # accept pd.Series as y
   pip install "pymlt[examples]"   # run the vignettes (lifelines, jupyter, …)
   pip install "pymlt[dev]"        # tests, linters, type checker
   pip install "pymlt[docs]"       # build the documentation locally

Development install
-------------------

.. code-block:: bash

   git clone https://github.com/RMKruse/pymlt
   cd pymlt
   pip install -e ".[dev,examples]"

Building the docs locally also requires `pandoc <https://pandoc.org/>`_,
which nbsphinx uses to execute and render the Jupyter vignettes:

.. code-block:: bash

   pip install -e ".[docs]"
   sphinx-build -b html -W --keep-going docs/ docs/_build/html
