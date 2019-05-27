#!/bin/sh

openie_output='../../code/outputs/openie/hashtag_emo-openie.op'
output='../../code/outputs/baseline/train_out_hashtag_emo.txt'

python3 emotion_cause_rule_extractor.py ${openie_output} ${output}