#!/usr/bin/env bash
# Reproduce every number and figure in the paper. CPU only; total runtime is roughly 1-2 hours.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
WORK="$ROOT/work"; mkdir -p "$WORK/res" "$WORK/fig"
cp "$ROOT"/src/*.py "$WORK/"
if [ -n "${SOURCE_DATA:-}" ]; then
  # Optional: rebuild the combined table from the original per-pathway tables
  (cd "$WORK" && python build.py "$SOURCE_DATA" all_raw.csv)
else
  cp "$ROOT/data/kegg_relations_combined.csv" "$WORK/all_raw.csv"
fi
cd "$WORK"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
python audit.py        | tee res/audit_log.txt     # structural audit (Table I)
python exp_row.py      | tee res/row_log.txt       # Task R, row-level (Table II)
python diag.py         | tee res/diag_log.txt      # per-row diagnosis
python exp.py          | tee res/log.txt           # Task M: pair, gene, pathway splits
python exp_fam.py      | tee res/fam_log.txt       # Task M: family-disjoint split
python exp_ai.py       | tee res/ai_log.txt        # CharCNN and GraphSAGE GNN (PyTorch)
python exp_more.py     | tee res/more_log.txt      # strata, learning curve, agreement, n-grams, tests
python exp_extra.py    | tee res/extra_log.txt     # stress tests: scrambling, swap, calibration, overlap
python exp_plus.py     | tee res/plus_log.txt      # ensemble, AUPRC, conformal coverage, precision@10, permutation control
python make_splits.py                               # export fold assignments
for f in fig1 fig_audit fig_res fig_more fig_extra fig_plus; do python $f.py; done
echo "Done. Results in $WORK/res, figures in $WORK/fig"
