#!/bin/sh

train_tb_parser_output='../../outputs/tb_parser/filtered_tweets_train.out'
train_output='../../outputs/BECR/train_out.txt'

time python bootstrap_rules.py ${train_tb_parser_output} ${train_output}
