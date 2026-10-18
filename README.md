# KEGG-Shortcut-Benchmark

Code, data and results for the paper

> **Gene Name Shortcuts Inflate AI and Language Model Benchmarks of Pathway Knowledge**
> Muhammad Azam and Michael Olaolu Arowolo (manuscript under review)

KEGG pathway relations (activation, inhibition, phosphorylation) are widely used as ground truth for testing whether AI systems and large language models "know" pathway biology. This repository shows that the standard benchmark design rewards shortcuts, and provides leakage-aware splits, baselines and stress tests to evaluate such systems properly.

## Key results

| Model (Task M, macro AUROC) | Pair-disjoint | Gene-disjoint | Family-disjoint | Pathway held out |
|---|---|---|---|---|
| Lookup (node profiles) | 0.974 | 0.891 | 0.785 | 0.597 |
| LR-Lex (gene symbols only) | 0.978 | 0.939 | 0.811 | 0.683 |
| GBDT-Graph | 0.987 | 0.878 | 0.776 | 0.606 |
| CharCNN | 0.986 | 0.929 | 0.771 | 0.697 |
| GraphSAGE GNN | 0.987 | 0.895 | 0.779 | 0.674 |
| BioBERT v1.1, frozen | 0.985 | 0.935 | 0.798 | 0.707 |
| BioBERT v1.0, frozen | 0.986 | 0.937 | 0.798 | 0.698 |
| SciBERT, frozen | 0.986 | 0.935 | 0.816 | 0.706 |
| SPECTER, frozen | 0.979 | 0.927 | 0.808 | 0.697 |
| BioBERT v1.1, fine-tuned | 0.981 | 0.937 | 0.831 | 0.732 |

* 19.6% of gene pairs carry more than one label; 41.0% of random-split test rows share their pair with training.
* A classifier that reads only the two gene symbols reaches macro-F1 0.731 under the random-row protocol.
* Phosphorylation transfers to unseen pathways (AUROC 0.87); activation/inhibition sign falls to 0.56-0.65.
* Split-conformal prediction (nominal 90%) covers 91% on pair-disjoint data but only 58% on unseen pathways.
* Ensembling three models helps only where shortcuts exist; a label-permutation control scores 0.49-0.52 under every split.
* Four open biomedical language models (BioBERT v1.1/v1.0, SciBERT, SPECTER) used as frozen encoders only match the symbol-only baseline; fine-tuned BioBERT gains modestly on held-out pathways (0.732 vs 0.683, Wilcoxon P = 0.046) while activation/inhibition stay at 0.68/0.66. Scrambling gene names costs the language models 0.20-0.23 AUROC on unseen genes.
* Stress tests: scrambling symbol spelling costs 0.09 AUROC on unseen genes; swapping starter and receiver drops pair-disjoint AUROC from 0.978 to 0.655; calibration error more than doubles on held-out pathways.

## Repository layout

```
data/
  kegg_relations_combined.csv        3,207 relations from 23 human KEGG pathways (starter, receiver, relation, pathway)
  splits/pairs_multilabel.csv        2,176 unique directed pairs, 3 binary labels, pathways, pair-disjoint folds
  splits/gene_and_family_folds.csv   860 genes, 432 name families, gene- and family-disjoint folds
src/                                 analysis and figure scripts (see run_all.sh for the order)
results/                             per-fold result tables and logs behind every number in the paper
figures/                             figures as used in the paper (PDF)
paper/KEGG-Shortcut-Benchmark_paper.pdf   preprint of the paper (submitted version)
```

Splits: pair-disjoint and gene-/family-disjoint use 5 folds x 2 seeds (columns `*_seed0`, `*_seed1`). For gene and family splits a pair is tested if either gene (family) is in the held-out fold and used for training only if neither is. Pathway-held-out holds out each pathway in turn and removes all of its pairs, in either orientation, from training.

## Reproduce

```bash
pip install -r requirements.txt
bash run_all.sh                          # uses data/kegg_relations_combined.csv
# or rebuild the table from the original per-pathway tables:
SOURCE_DATA=/path/to/pathway_tables bash run_all.sh
```

Everything runs on a CPU. Outputs go to `work/res` and `work/fig`. `src/lm_setup.sh` downloads the four language models from the authors' public releases (GitHub and AllenAI S3) and converts them to PyTorch; BioBERT fine-tuning takes several hours on a CPU.

## Source data

The relation tables were introduced in M. Azam *et al.*, Research Square preprint, doi:10.21203/rs.3.rs-9243236/v1. The combined table in `data/` is the version used in this study.
KEGG content is subject to the KEGG terms of use (https://www.kegg.jp/kegg/legal.html).

## Citation

See `CITATION.cff`. Citation details will be updated on publication.

## Licence

MIT (see `LICENSE`).
