#!/bin/sh

# baseline 2 - rules over dependency output
tb_parser_output='../../outputs/tb_parser/tweets_small_sample_set.out'
output='../../outputs/baseline/tb_test_small.txt'

python3 dependency_rule_extractor.py ${tb_parser_output} ${output}
