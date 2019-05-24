#!/bin/sh

#dm_lexicon='../lib/emotion_lexicon/DepecheMood/DepecheMood_freq.txt'
dm_lexicon='../lib/emotion_lexicon/DepecheMood/DepecheMood_normfreq.txt'
nrc_lexicon='../lib/emotion_lexicon/NRC/NRC-Emotion-Lexicon-Wordlevel-v0.92.txt'
kw_lexicon='../lib/emotion_lexicon/emotion_kw_list/emotion_keywords.txt'

python3 build_lexicon.py ${dm_lexicon} 'dm'
python3 build_lexicon.py ${nrc_lexicon} 'nrc'
python3 build_lexicon.py ${kw_lexicon} 'kw'
python3 build_lexicon.py ${kw_lexicon} 'kwsyns'
