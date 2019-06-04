#!/bin/sh

# baseline 2 - rules over dependency output
tb_parser_output='../../outputs/tb_parser/filtered_tweets_train.out'
output='../../outputs/BECR/emph_train_out_no_neg.txt'

time python bootstrap_rules.py ${tb_parser_output} ${output}
