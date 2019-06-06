#!/bin/sh

input='../data/preprocessed/split/filtered_tweets_test.txt'
output='../results/baseline/recall_filtered_test_25.txt'

python make_recall_file.py ${input} ${output}