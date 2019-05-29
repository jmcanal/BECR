#!/bin/sh

train_tok='../data/preprocessed/annotated/train_sources.tok'
train_tag='../data/preprocessed/annotated/train_targets.txt'
test_tok='../data/preprocessed/annotated/dev_sources.tok'
test_tag='../data/preprocessed/annotated/dev_targets.txt'
acc_file='../results/tagger/accuracy.txt'
pred_file='../results/tagger/predictions.txt'

python run_tagger.py ${train_tok} ${train_tag} ${test_tok} ${test_tag} ${acc_file} ${pred_file}