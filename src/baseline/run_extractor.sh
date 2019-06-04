#!/bin/sh

# baseline 1 - rules over openie output
openie_output='../../outputs/openie/hashtag_emo-openie.op'
output='../../outputs/baseline/train_out_hashtag_emo.txt'

#python3 openie_rule_extractor.py ${openie_output} ${output}

# baseline 2 - rules over dependency output
tb_parser_output='../../outputs/tb_parser/filtered_tweets_test.out'
output='../../outputs/baseline/tb_test.txt'

python3 dependency_rule_extractor.py ${tb_parser_output} ${output}
