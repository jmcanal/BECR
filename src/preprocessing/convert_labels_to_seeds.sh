#!/bin/sh

# given tokenized tweets and labels, create seed data
tweet_tokens='../../data/preprocessed/annotated/tagged_train_sample.tok'
tweet_labels='../../data/preprocessed/annotated/labeled_train_sample.txt'

/opt/python-3.6/bin/python3.6 convert_labels_to_seeds.py ${tweet_tokens} ${tweet_labels}
