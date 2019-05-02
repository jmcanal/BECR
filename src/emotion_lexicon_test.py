"""
Test to see whether emotion lexicons identify usable emotion vocabulary
to extract from tweets
"""

import sys
from scipy.sparse import dok_matrix, vstack
from collections import defaultdict as dd
import string
import numpy as np
np.set_printoptions(threshold=sys.maxsize)
import pickle

lexicon_file = sys.argv[1]
# tweet_file = sys.argv[2]
output = sys.argv[2]

# The NRC (National Research Council Canada) emotion lexicon

class EmoLexicon():
    emotion_lexicon = dok_matrix((0,0))
    emotion_to_idx = {'anger': 0, 'anticipation': 1, 'disgust': 2,
                      'fear': 3, 'joy': 4, 'negative': 5, 'positive': 6,
                      'sadness': 7, 'surprise': 8, 'trust': 9}
    curr_emo_array = [0 for n in emotion_to_idx.values()]
    word_to_idx = dict()

    def __init__(self, lexicon_file, output):
        self.lexicon_file = lexicon_file
        self.output = output

    def read_file(self):
        with open(lexicon_file, "r") as lex_file:
            index = 0
            for line in lex_file:
                if not line.startswith('\n'):
                    word, emotion, value = line.split()
                    emo_index = self.emotion_to_idx[emotion]
                    value = int(value)
                    if word in self.word_to_idx.keys():
                        self.curr_emo_array[emo_index] = value
                        # self.emotion_lexicon[index, emo_index] = value
                    else:
                        self.emotion_lexicon = vstack([self.emotion_lexicon, self.curr_emo_array])
                        self.curr_emo_array = [0 for n in self.emotion_to_idx.values()]
                        index += 1
                        self.word_to_idx[word] = index
                        # print(index, emo_index)
                        # print(self.curr_emo_array)
                        self.curr_emo_array[emo_index] = value
        # with open(self.output, "w") as out:
        #     out.write(self.emotion_lexicon.toarray())
        numpy_emo_array = self.emotion_lexicon.toarray()
        np.save(self.output, numpy_emo_array)
        with open('/Users/jmcanal/Dropbox (Professional Serv)/CLMS/Ling_575_IE/project/zoolia/lib/emotion_dictionary.pkl', 'wb') as dict_out:
            pickle.dump(self.word_to_idx, dict_out)
        # print(numpy_emo_array[self.word_to_idx['zip']])


EmoLexicon(lexicon_file, output).read_file()
