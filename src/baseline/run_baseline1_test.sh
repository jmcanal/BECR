#!/bin/sh

# baseline 1 - rules over openie output
openie_output='../../outputs/openie/test_openie.op'
output='../../outputs/baseline/openie_test.txt'

/opt/python-3.6/bin/python3.6 openie_rule_extractor.py ${openie_output} ${output}

