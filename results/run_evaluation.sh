#!/bin/sh

ec_file='../outputs/BECR/test_out.txt'
gold_file='BECR/25_recall.txt'
precision_file_10='BECR/top_10_precision.txt'
precision_file_25='BECR/top_25_precision.txt'
output_file_10='BECR/10_results'
output_file_25='BECR/25_results'

python evaluate.py ${ec_file} ${gold_file} ${precision_file_10} ${output_file_10} 10
python evaluate.py ${ec_file} ${gold_file} ${precision_file_25} ${output_file_25} 25

#ec_file='../outputs/baseline/tb_test.txt'
#gold_file='baseline/25_recall.txt'
#precision_file='baseline/precision_baseline.txt'
#output_file='baseline/results'
#
#python evaluate.py ${ec_file} ${gold_file} ${precision_file} ${output_file} 25