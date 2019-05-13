#!/bin/sh

data_file='../data/preprocessed/filtered/filtered_tweets_total.txt'
train_file='../data/preprocessed/filtered/filtered_tweets_train.txt'
dev_file='../data/preprocessed/filtered/filtered_tweets_dev.txt'
devtest_file='../data/preprocessed/filtered/filtered_tweets_devtest.txt'
test_file='../data/preprocessed/filtered/filtered_tweets_test.txt'

python3 split_data.py ${data_file} ${train_file} ${dev_file} ${devtest_file} ${test_file}