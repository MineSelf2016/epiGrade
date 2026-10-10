#!/bin/bash
cd "$(dirname "$0")/.."
for f in make_figures fig_overview fig_performance fig_mechanism fig_reliability fig_biology fig_grade_transfer fig_supplementary; do
  echo "== $f"; .venv/bin/python -u code/experiments/$f.py 2>&1 | tail -n 6
done
echo "== check"; .venv/bin/python -u code/scripts/check_figures.py 2>&1 | tail -n 15
