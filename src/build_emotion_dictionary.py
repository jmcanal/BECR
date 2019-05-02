"""
Identify usable emotion vocabulary from emotion lexicons to extract from tweets
"""

import sys
from scipy.sparse import dok_matrix, vstack
import numpy as np
np.set_printoptions(threshold=sys.maxsize)
import pickle


class EmoLexicon:
    emotion_lexicon = dok_matrix((0,0), dtype=float)
    emo2idx = {}
    w2idx = {}

    def __init__(self, lexicon_file):
        self.lexicon = self.setup(lexicon_file)
        self.lib_path = '../lib/emotion_lexicon/'

    def setup(self, lexicon_file):
        lines = [line for line in open(lexicon_file, "r")]
        for item in lines[0].split("\t")[1:]:
            self.emo2idx[item] = len(self.emo2idx.keys())

        return lines[1:]

    def read_file(self):
        for line in self.lexicon:
            line_list = line.split()
            word = line_list[0]
            emotion_list = np.array(line_list[1:], dtype=float)

            self.emotion_lexicon = vstack([self.emotion_lexicon, emotion_list])
            self.w2idx[word] = len(self.w2idx.keys())

        numpy_emo_array = self.emotion_lexicon.toarray()

        # np.save(self.lib_path + 'NRC/nrc_emotion_lexicon_matrix.npy', numpy_emo_array)
        # pickle.dump(self.w2idx, open(self.lib_path + 'NRC/nrc_word_map.pkl', 'wb'))
        # pickle.dump(self.emo2idx, open(self.lib_path + 'NRC/nrc_emotion_map.pkl', 'wb'))

        np.save(self.lib_path + 'DepecheMood/dm_emotion_lexicon_matrix.npy', numpy_emo_array)
        pickle.dump(self.w2idx, open(self.lib_path + 'DepecheMood/dm_word_map.pkl', 'wb'))
        pickle.dump(self.emo2idx, open(self.lib_path + 'DepecheMood/dm_emotion_map.pkl', 'wb'))


def main():
    """
    Create emotion lexicon matrix from lexicon
    :return:
    """

    lexicon_file = sys.argv[1]
    EmoLexicon(lexicon_file).read_file()


if __name__ == "__main__":
    main()
