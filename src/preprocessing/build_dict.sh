#!/bin/sh

# code to run build other dictionaries we aren't using
#dm_lexicon='../lib/emotion_lexicon/DepecheMood/DepecheMood_freq.txt'
#dm_norm_lexicon='../lib/emotion_lexicon/DepecheMood/DepecheMood_normfreq.txt'
#nrc_lexicon='../lib/emotion_lexicon/NRC/NRC-Emotion-Lexicon-Wordlevel-v0.92.txt'
#python3 build_lexicon.py ${dm_lexicon} 'dm'
#python3 build_lexicon.py ${dm_norm_lexicon} 'dmnorm'
#python3 build_lexicon.py ${nrc_lexicon} 'nrc'
#python3 build_lexicon.py ${kw_lexicon} 'kwsyns'

# build the keyword lexicon
kw_lexicon='../../lib/emotion_lexicon/emotion_kw_list/emotion_keywords.txt'
/opt/python-3.6/bin/python3.6 build_lexicon.py ${kw_lexicon} 'kw'
