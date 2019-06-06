#!/bin/sh

# baseline 1 - rules over openie output
openie_output='../../outputs/openie/hashtag_emo-openie.op'
output='../../outputs/baseline/train_out_hashtag_emo.txt'

python3 openie_rule_extractor.py ${openie_output} ${output}

