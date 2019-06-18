# BECR 
BECR -- Bootstrapping Emotion Cause Relations -- is a semi-supervised algorithm for extracting emotion-cause relations from Tweets.

## Directory: data

Includes text files of Tweets, both in their original formats (semeval files) as well as the files that were aggregatd and randomly split into train, dev, devtest and test files. 

Most files contain unlabeled Tweets, but the `annotated` folder contains Tweets that we hand-labeled. The hand-labeled Tweets were used in the neural methods and as seeds for BECR algorithm.

## Directory: lib

Contains external libraries, dictionaries and keyword lists used in the different phases of the project. The library also contains pickle files from Seeds in the training phase. The `test_seeds.pkl` file is used as the "model" in the test phase of BECR.

The TweeboParser dependency parser and the GloVe Twitter embedings are not included in this package because of their large file sizes. See http://www.cs.cmu.edu/~ark/TweetNLP/ for the TweeboParser and https://nlp.stanford.edu/projects/glove/ for GloVe Twitter embeddings. 

## Directory: outputs

Output files for baseline, experimental and final systems. Note that the BECR algorithm uses the tb_parser (TweeboParser) outputs as its input.

## Directory: results

The evaluation results for the systems. Included are instructions and files for running the evaluations scripts on patas.

## Directory: src

The source code for the baseline, experimental and final BECR systems. Included are instructions for running the scripts on patas.
