"""
Process GLoVe .txt files and create pickle files for embedding dictionaries
"""

import sys
import pickle

GLOVE_SIZE = 25


class GloveVectors:

    def __init__(self, file):
        self.file = file
        self.glove_embeddings = {}
        self.lib_path = '../lib/glove/'

    def make_embeddings(self):
        with open(self.file, 'r') as f:
            for line in f:
                line = line.split()
                word = line[0]
                embedding = [float(n) for n in line[1:]]
                self.glove_embeddings[word] = embedding
        # return self.glove_embeddings

    def save_embeddings(self):
        self.make_embeddings()
        # print(self.glove_embeddings)
        filename = "glove" + str(GLOVE_SIZE) + ".pkl"
        pickle.dump(self.glove_embeddings, open(self.lib_path + filename, 'wb'))


def main():
    """
    Apply rules to Tweeboparser outputs and return emotion-cause relations when found for tweets
    :return: void
    """
    glove = sys.argv[1]
    GloveVectors(glove).save_embeddings()


if __name__ == "__main__":
    main()