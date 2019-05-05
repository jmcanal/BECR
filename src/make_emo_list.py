"""
Create an emotion keyword list from various lists online
Expanded using WordNet synsets
"""


import pickle
from nltk.corpus import wordnet as wn


def create_emo_list(file):
    emo_set = set()
    # delete_list = ['might']
    with open(file, "r") as f:
        for line in f:
            line = line.split()
            if not len(line) > 1:
                word = line[0].lower()
                emo_set.add(word)
                syn = wn.synsets(word)
                for lemmas in syn:
                    pos = ('v', 'a')
                    if lemmas.pos() is pos:
                        for lemma in lemmas.lemma_names():
                            if not "_" in lemma:
                                emo_set.add(lemma.lower())

    # for item in delete_list:
    #     emo_set.remove(item)
    print(len(emo_set))
    return emo_set
    # print(emo_set)



def main():
    """
    Create emotion list from text file
    :return:
    """
    raw_emo_list = '../lib/emotion_lexicon/emotion_kw_list/emotion_keywords.txt'
    emo_set = create_emo_list(raw_emo_list)
    pickle.dump(emo_set, open('../lib/emotion_lexicon/emotion_kw_list/emotion_keywords.pkl', 'wb'))


if __name__ == "__main__":
    main()