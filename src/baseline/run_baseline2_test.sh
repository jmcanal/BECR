#!/bin/sh

# baseline 2 - rules over dependency output
tb_parser_output='../../outputs/tb_parser/filtered_tweets_test.out'
output='../../outputs/baseline/tb_test.txt'

python3 dependency_rule_extractor.py ${tb_parser_output} ${output}
