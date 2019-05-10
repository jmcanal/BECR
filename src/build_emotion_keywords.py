"""
Create an emotion keyword list from various lists online
Expanded using WordNet synsets
"""

import pickle
from nltk.corpus import wordnet as wn

emo_set = set()

def create_emo_list(keyword_file):
    """
    Create a list of emotion keywords by cleaning the given list and expanding using synsets from Wordnet
    :param keyword_file: the given file of emotion keywords
    :return: void
    """
    keywords = [line.strip("\n") for line in open(keyword_file, "r")]
    for kw in keywords:
        kw = kw.split()
        if not len(kw) > 1:
            word = kw[0].lower()
            emo_set.add(word)
            add_syns(word)


def add_syns(word):
    """
    Add words from Wordnet synsets
    :param word:
    :return: void
    """
    global emo_set
    syns = wn.synsets(word)
    va_syns = [syn for syn in syns if syn.pos() in ('v', 'a')]
    for syn in va_syns:
        lemmas = set([lemma.lower() for lemma in syn.lemma_names() if not "_" in lemma])
        emo_set = emo_set.union(lemmas)


def main():
    """
    Create emotion list from text file
    :return:
    """
    raw_emo_list = '../lib/emotion_lexicon/emotion_kw_list/emotion_keywords.txt'
    create_emo_list(raw_emo_list)
    with open('../lib/emotion_lexicon/emotion_kw_list/emotion_keyword_syns_list.txt', 'w') as f:
        print("\n".join(emo_set), file=f)
    pickle.dump(emo_set, open('../lib/emotion_lexicon/emotion_kw_list/emotion_keywords_syns.pkl', 'wb'))


if __name__ == "__main__":
    main()