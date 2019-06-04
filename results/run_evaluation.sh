#!/bin/sh

ec_file='../outputs/baseline/tb_test.txt'
gold_file='baseline/recall_filtered_test_25.txt'
precision_file='baseline/precision_baseline.txt'
output_file='baseline/recall_baseline.txt'

python3 evaluate.py ${ec_file} ${gold_file} ${precision_file} ${output_file}