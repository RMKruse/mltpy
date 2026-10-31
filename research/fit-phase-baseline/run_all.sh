#!/bin/zsh
P=/Users/entropy/Git/mltpy/.venv/bin/python
cd ${0:A:h}
for c in $($P harness.py --list --case x --out x); do
  s=$(date +%s)
  if $P harness.py --case $c --out results/$c.json > logs/$c.log 2>&1; then st=ok; else st=FAIL; fi
  echo "$c $st $(( $(date +%s) - s ))s" >> logs/summary.txt
done
echo DONE >> logs/summary.txt
