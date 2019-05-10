#!/bin/sh

original='../data/electoraltweets/batch1.txt'
raw_tweets='../data/preprocessed/raw/electoral-batch1-raw.txt'
output_file='../data/preprocessed/filtered/filtered_tweets_labeled_kwsyns.txt'
debug_file='../data/preprocessed/debug/labeled_debug_kwsyns.txt'

python3 filter_by_emotion.py ${original} ${raw_tweets} ${output_file} ${debug_file}