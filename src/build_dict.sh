#!/bin/sh

lexicon_file='../lib/emotion_lexicon/DepecheMood/DepecheMood_normfreq.txt'

python3 build_emotion_dictionary.py ${lexicon_file}