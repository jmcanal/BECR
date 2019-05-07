#!/bin/sh

raw_tweets='../data/preprocessed/hashtag-raw.txt'
output_file='../data/preprocessed/hashtag-emotweets.txt'

python3 tweet_emotion_filter.py ${raw_tweets} ${output_file}