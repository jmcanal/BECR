"""
Process GLoVe .txt files and create pickle files for embedding dictionaries
"""

import sys
import pickle


class GloveVectors:

    GLOVE_SIZE = 200

    def __init__(self, file):
        self.file = file
        self.glove_embeddings = {}
        self.lib_path = '../lib/glove/'

    def make_embeddings(self):
        """
        Make the embeddings dictionary
        :return: void
        """
        with open(self.file, 'r') as f:
            for line in f:
                line = line.split()
                word = line[0]
                embedding = [float(n) for n in line[1:]]
                self.glove_embeddings[word] = embedding

    def save_embeddings(self):
        """
        Pickle the glove embedding dictionary
        :return: void
        """
        self.make_embeddings()
        filename = "glove" + str(self.GLOVE_SIZE) + ".pkl"
        pickle.dump(self.glove_embeddings, open(self.lib_path + filename, 'wb'))


def main():
    """
    Pickle the GloVe Embeddings
    :return: void
    """
    glove = sys.argv[1]
    GloveVectors(glove).save_embeddings()


if __name__ == "__main__":
    main()
