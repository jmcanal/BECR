# BECR

Instructions for running BECR test, small test or train, are below. It is assumed that the command will be entered _within_ the current directory (`../zoe_julia/src/BECR`)

# full test

Recommended method for running: submit a job to condor (the run has been taking 10-20 minutes on condor; ~5 minute on our local machines):

`condor_submit run_test.cmd`

To run the bash script directly (not recommended):

`./run_test.sh`

Alternatively, to run directly from the command line, bypassing the bash script, run the following in one line (not recommended):

`/opt/python-3.6/bin/python3.6 bootstrap_rules.py ../../outputs/tb_parser/filtered_tweets_test.out ../../outputs/BECR/test_out.txt`

# small test

Recommended method for running: submit a job to condor (the run took < 1 minute on condor; < 1 minute on our local machines):

`condor_submit run_test_small.cmd`

To run the bash script directly (not recommended):

`./run_test_small.sh`

Alternatively, to run directly from the command line, bypassing the bash script, run the following in one line (not recommended):

`/opt/python-3.6/bin/python3.6 bootstrap_rules.py ../../outputs/tb_parser/tweets_small_sample_set.out ../../outputs/BECR/test_small_out.txt`

# train

Recommended method for running: submit a job to condor (the run has been taking 10-20 minutes on condor; 5-10 minutes on our local machines):

`condor_submit run_train.cmd`

To run the bash script directly (not recommended):

`./run_train.sh`

Alternatively, to run directly from the command line, bypassing the bash script, run the following in one line (not recommended):

`/opt/python-3.6/bin/python3.6 bootstrap_rules.py ../../outputs/tb_parser/filtered_tweets_train.out ../../outputs/BECR/train_out.txt`