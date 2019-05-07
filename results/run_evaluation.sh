#!/bin/sh

ec_file='../../code/outputs/baseline/train_out_hashtag_emo.txt'
gold_file='../../code/data/preprocessed/hashtag-emo_gold.txt'
precision_file='../../code/results/precision_hashtag_emo.txt'
output_file='../../code/results/hashtag_emo_results.txt'

python3 evaluate.py ${ec_file} ${gold_file} ${precision_file} ${output_file}