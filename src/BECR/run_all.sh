#!/bin/sh

train_tb_parser_output='../../outputs/tb_parser/filtered_tweets_train.out'
train_output='../../outputs/BECR/train_out.txt'
test_tb_parser_output='../../outputs/tb_parser/filtered_tweets_test.out'
test_output='../../outputs/BECR/test_out.txt'

time /opt/python-3.6/bin/python3.6 bootstrap_rules.py ${train_tb_parser_output} ${train_output}
time /opt/python-3.6/bin/python3.6 bootstrap_rules.py ${test_tb_parser_output} ${test_output} --test
