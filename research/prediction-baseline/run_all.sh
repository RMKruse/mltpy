#!/bin/sh
# Run every case sequentially (one process each) so timings do not contend.
cd "$(dirname "$0")"
PY=${PY:-../../../../../.venv/bin/python}
for c in ${CASES:-ia_bs3x1_m8_n1000 boxcox_x3 boxcox_scaled right_baseline coxph_x3 survreg_x3 ia_bs4x2_m15_n1000 ia_bs6x3_m28_n250 ia_scaled_bs3x1_m9_n1000}; do
  echo "== $c $(date +%T)"
  "$PY" harness.py --case "$c" --out "results/$c.json" 2>&1 | tail -3
done
echo "ALLDONE $(date +%T)"
