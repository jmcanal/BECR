#!/bin/sh

#original='../data/hashtagemotion/Jan9-2012-tweets-clean.txt'
raw_tweets='../data/preprocessed/raw/semeval-2018-all-raw.txt'
output_file='../data/preprocessed/filtered/filtered_tweets_total.txt'
debug_file='../data/preprocessed/debug/total_debug.txt'

python3 filter_by_emotion.py ${raw_tweets} ${output_file} ${debug_file}