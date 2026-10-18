#!/usr/bin/env bash
# Reproduce every number and figure in the paper. CPU only; total runtime is roughly 4-6 hours, most of it for BioBERT fine-tuning.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
WORK="$ROOT/work"; mkdir -p "$WORK/res" "$WORK/fig"
cp "$ROOT"/src/*.py "$ROOT"/src/*.sh "$WORK/"
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
export HF_HUB_OFFLINE=1
bash lm_setup.sh                                    # download and convert BioBERT v1.1/v1.0, SciBERT, SPECTER
python exp_lm.py frozen lm/biobert_pt          | tee res/lm_frozen_log.txt
python exp_lm.py frozen lm/biobert10_pt _biobert10 | tee res/lm_frozen_biobert10_log.txt
python exp_lm.py frozen lm/scibert_pt _scibert     | tee res/lm_frozen_scibert_log.txt
python exp_lm.py frozen lm/specter_pt _specter     | tee res/lm_frozen_specter_log.txt
python exp_lm.py ft lm/biobert_pt                  | tee res/lm_ft_log.txt     # fine-tuning (several hours on CPU)
python lm_stats.py
python make_splits.py                               # export fold assignments
for f in fig1 fig_audit fig_res fig_more fig_extra fig_plus fig_lm; do python $f.py; done
echo "Done. Results in $WORK/res, figures in $WORK/fig"
