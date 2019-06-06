#!/bin/sh

test_tb_parser_output='../../outputs/tb_parser/filtered_tweets_test.out'
test_output='../../outputs/BECR/test_out.txt'

time python bootstrap_rules.py ${test_tb_parser_output} ${test_output} --test
