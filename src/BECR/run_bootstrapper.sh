#!/bin/sh

# baseline 2 - rules over dependency output
train_tb_parser_output='../../outputs/tb_parser/filtered_tweets_train.out'
#train_tb_parser_output='../../outputs/tb_parser/tweets_small_sample_set.out'
train_output='../../outputs/BECR/train_out.txt'
test_tb_parser_output='../../outputs/tb_parser/filtered_tweets_test.out'
test_output='../../outputs/BECR/test_out.txt'
#test_tb_parser_output='../../outputs/tb_parser/tweets_small_sample_set.out'

time python bootstrap_rules.py ${train_tb_parser_output} ${train_output}
time python bootstrap_rules.py ${test_tb_parser_output} ${test_output} --test
