# Evaluation

Instructions for running evaluation on the BECR and baseline 2 extractors are below. It is assumed that the command will be entered _within_ the current directory (`../zoe_julia/results`)

All scripts run in a few seconds on both patas and our local machines.

The performance summaries, or evaluation outputs, are printed to text files labeled results in the relevant directories.


## BECR evaluation

To run the bash script directly:

`./run_becr_evaluation.sh`

Alternatively, to run directly from the command line, bypassing the bash script, run the following two lines:

`/opt/python-3.6/bin/python3.6 evaluate.py ../outputs/BECR/test_out.txt recall_file.txt BECR/top_10_precision.txt BECR/10_results 10`

`/opt/python-3.6/bin/python3.6 evaluate.py ../outputs/BECR/test_out.txt recall_file.txt BECR/top_25_precision.txt BECR/25_results 25`

For output files, see directory `../zoe_julia/results/BECR`

Files:
- 10_results
- 25_results
- 10_results_relaxed
- 25_results_relaxed


## Baseline evaluation

To run the bash script directly:

`./run_baseline_evaluation.sh`

Alternatively, to run directly from the command line, bypassing the bash script, run the following in one line:

`/opt/python-3.6/bin/python3.6 evaluate.py ../outputs/baseline/tb_test.txt recall_file.txt baseline/precision_baseline.txt baseline/results 25`

For output files, see directory `../zoe_julia/results/baseline`

Files:
- results
- results_relaxed
