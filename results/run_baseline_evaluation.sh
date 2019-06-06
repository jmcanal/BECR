#!/bin/sh

ec_output_file='../outputs/baseline/tb_test.txt'
recall_file='recall_file.txt'
precision_file='baseline/precision_baseline.txt'
output_file='baseline/results'

time /opt/python-3.6/bin/python3.6 evaluate.py ${ec_output_file} ${recall_file} ${precision_file} ${output_file} 25