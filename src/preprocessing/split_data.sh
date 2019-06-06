#!/bin/sh

data_file='../data/preprocessed/filtered/kw_tweets.out'
train_tok='../data/preprocessed/annotated/tagged_train_sample.tok'
train_file='../data/preprocessed/split/filtered_tweets_train.txt'
dev_file='../data/preprocessed/split/filtered_tweets_dev.txt'
devtest_file='../data/preprocessed/split/filtered_tweets_devtest.txt'
test_file='../data/preprocessed/split/filtered_tweets_test.txt'

# split the data into test/train
/opt/python-3.6/bin/python3.6 split_data.py ${data_file} ${train_tok} ${train_file} ${dev_file} ${devtest_file} ${test_file}