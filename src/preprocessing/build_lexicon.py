"""
Build dictionaries of emotions and emotion words from the given lexicon
"""

import sys
from scipy.sparse import dok_matrix, vstack
from nltk.corpus import wordnet as wn
import numpy as np
np.set_printoptions(threshold=sys.maxsize)
import pickle


class EmoLexicon:

    emotion_lexicon = dok_matrix((0,0), dtype=float)
    emo2idx = {}
    w2idx = {}
    emo_set = set()

    KW = ('kwsyns', 'kw')
    DICT = ('nrc', 'dm')

    def __init__(self, lexicon_file, lexicon_type):
        """
        Initialize this class with the lexicon file
        :param lexicon_file: the given emotion lexicon
        """
        self.lexicon_type = lexicon_type
        self.lexicon = self.setup(lexicon_file)
        self.lib_path = '../../lib/emotion_lexicon/'

    def setup(self, lexicon_file):
        """
        Set up the emotion to index mapping and strip newlines from lexicon lines
        :param lexicon_file: the given emotion lexicon
        :return: the lines from the lexicon without the headers
        """
        lines = [line.strip("\n") for line in open(lexicon_file, "r")]

        if self.lexicon_type in self.KW:
            return lines

        # use the headers to set up possible emotion labels
        for item in lines[0].split("\t")[1:]:
            self.emo2idx[item] = len(self.emo2idx.keys())

        return lines[1:]

    def read_file(self):
        """
        Read in the lexicon and create the emotion matrix and the mappings
        Creates pickles for both maps, and outputs numpy matrix
        :return: void
        """
        if self.lexicon_type in self.DICT:
            for line in self.lexicon:
                line_list = line.split()
                word = line_list[0]
                emotion_list = np.array(line_list[1:], dtype=float)

                self.emotion_lexicon = vstack([self.emotion_lexicon, emotion_list])
                self.w2idx[word] = len(self.w2idx.keys())

        elif self.lexicon_type in self.KW:
            for kw in self.lexicon:
                kw = kw.split()
                if not len(kw) > 1:
                    word = kw[0].lower()
                    self.emo_set.add(word)
                    if self.lexicon_type == 'kwsyns':
                        self.add_syns(word)

    def add_syns(self, word):
        """
        Add words from Wordnet synsets
        :param word: the given word
        :return: void
        """
        syns = wn.synsets(word)
        va_syns = [syn for syn in syns if syn.pos() in ('v', 'a')]
        for syn in va_syns:
            lemmas = set([lemma.lower() for lemma in syn.lemma_names() if not "_" in lemma])
            self.emo_set = self.emo_set.union(lemmas)

    def pickle_info(self):
        """
        Pickle the data structures created for later use
        :return: void
        """
        numpy_emo_array = self.emotion_lexicon.toarray()

        if self.lexicon_type == 'nrc':
            np.save(self.lib_path + 'NRC/nrc_emotion_lexicon_matrix.npy', numpy_emo_array)
            pickle.dump(self.w2idx, open(self.lib_path + 'NRC/nrc_word_map.pkl', 'wb'))
            pickle.dump(self.emo2idx, open(self.lib_path + 'NRC/nrc_emotion_map.pkl', 'wb'))

        elif self.lexicon_type == 'dm':
            np.save(self.lib_path + 'DepecheMood/dm_emotion_lexicon_matrix.npy', numpy_emo_array)
            pickle.dump(self.w2idx, open(self.lib_path + 'DepecheMood/dm_word_map.pkl', 'wb'))
            pickle.dump(self.emo2idx, open(self.lib_path + 'DepecheMood/dm_emotion_map.pkl', 'wb'))

        elif self.lexicon_type == 'kwsyns':
            with open(self.lib_path + 'emotion_kw_list/emotion_keyword_syns_list.txt', 'w') as f:
                print("\n".join(self.emo_set), file=f)
            pickle.dump(self.emo_set, open(self.lib_path + 'emotion_kw_list/emotion_keywords_syns.pkl', 'wb'))

        elif self.lexicon_type == 'kw':
            with open(self.lib_path + 'emotion_kw_list/emotion_keyword_list.txt', 'w') as f:
                print("\n".join(self.emo_set), file=f)
            pickle.dump(self.emo_set, open(self.lib_path + 'emotion_kw_list/emotion_keywords.pkl', 'wb'))


def main():
    """
    Create emotion lexicon matricies and lists from lexicon
    :return:
    """
    lexicon_file = sys.argv[1]
    lexicon_type = sys.argv[2]
    lexicon = EmoLexicon(lexicon_file, lexicon_type)
    lexicon.read_file()
    lexicon.pickle_info()


if __name__ == "__main__":
    main()
