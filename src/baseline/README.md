# Baseline

Instructions to run the dependency rule extractor (baseline 2) test, small test or train, are below. There are also instructions for running the OpenIE (baseline 1) test. It is assumed that the command will be entered _within_ the current directory (`../zoe_julia/src/baseline`)

All scripts run in a few seconds for us on both patas and our local machines.

# full baseline 2 test

To run the bash script directly:

`./run_baseline2_test.sh`

Alternatively, to run directly from the command line, bypassing the bash script, run the following in one line:

`/opt/python-3.6/bin/python3.6 dependency_rule_extractor.py ../../outputs/tb_parser/filtered_tweets_test.out ../../outputs/baseline/tb_test.txt`

# small baseline 2 test

To run the bash script directly:

`./run_baseline2_test_small.sh`

Alternatively, to run directly from the command line, bypassing the bash script, run the following in one line:

`/opt/python-3.6/bin/python3.6 dependency_rule_extractor.py ../../outputs/tb_parser/tweets_small_sample_set ../../outputs/baseline/tb_test_small.txt`

# baseline 2 train

To run the bash script directly:

`./run_baseline2_train.sh`

Alternatively, to run directly from the command line, bypassing the bash script, run the following in one line:

`/opt/python-3.6/bin/python3.6 dependency_rule_extractor.py ../../outputs/tb_parser/filtered_tweets_train.out ../../outputs/baseline/tb_train.txt`

# baseline 1 test

To run the bash script directly:

`./run_baseline1_test.sh`

Alternatively, to run directly from the command line, bypassing the bash script, run the following in one line:

`/opt/python-3.6/bin/python3.6 openie_rule_extractor.py ../../outputs/openie/test_openie.op ../../outputs/baseline/openie_test.txt`