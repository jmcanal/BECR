#!/bin/sh

ec_output_file='../outputs/baseline/tb_test.txt'
recall_file='recall_file.txt'
precision_file='baseline/precision_baseline.txt'
output_file='baseline/results'

time python evaluate.py ${ec_output_file} ${recall_file} ${precision_file} ${output_file} 25