#!/usr/bin/env bash
# Download the four open biomedical language models from the authors' public releases and convert them to PyTorch.
# Creates lm/biobert_pt, lm/biobert10_pt, lm/scibert_pt and lm/specter_pt in the working directory.
set -euo pipefail
mkdir -p lm/dl_bb11 lm/dl_bb10 lm/dl_scibert lm/dl_specter
curl -sSL -o lm/dl_bb11/b.tar.gz https://github.com/naver/biobert-pretrained/releases/download/v1.1-pubmed/biobert_v1.1_pubmed.tar.gz
curl -sSL -o lm/dl_bb10/b.tar.gz https://github.com/naver/biobert-pretrained/releases/download/v1.0-pubmed-pmc/biobert_v1.0_pubmed_pmc.tar.gz
curl -sSL -o lm/dl_scibert/s.tar https://s3-us-west-2.amazonaws.com/ai2-s2-research/scibert/huggingface_pytorch/scibert_scivocab_uncased.tar
curl -sSL -o lm/dl_specter/a.tar.gz https://ai2-s2-research-public.s3-us-west-2.amazonaws.com/specter/archive.tar.gz
tar xzf lm/dl_bb11/b.tar.gz -C lm/dl_bb11; tar xzf lm/dl_bb10/b.tar.gz -C lm/dl_bb10
tar xf lm/dl_scibert/s.tar -C lm/dl_scibert; tar xzf lm/dl_specter/a.tar.gz -C lm/dl_specter
mkdir -p lm/dl_specter/m && tar xzf lm/dl_specter/model.tar.gz -C lm/dl_specter/m
# TensorFlow and PyTorch are run in separate processes (importing both in one process can crash)
python lm_dump_tf.py lm/dl_bb11/biobert_v1.1_pubmed/model.ckpt-1000000 lm/bb11.npz
python lm_tf_to_pt.py lm/dl_bb11/biobert_v1.1_pubmed lm/bb11.npz lm/biobert_pt
python lm_dump_tf.py lm/dl_bb10/biobert_v1.0_pubmed_pmc/biobert_model.ckpt lm/bb10.npz
python lm_tf_to_pt.py lm/dl_bb10/biobert_v1.0_pubmed_pmc lm/bb10.npz lm/biobert10_pt
python lm_scibert_specter.py lm/dl_scibert/scibert_scivocab_uncased lm/scibert_pt lm/dl_specter/m/weights.th lm/specter_pt
