"""
A small script to check GLoVe scores
"""

import numpy as np
import pickle
from scipy import spatial

WORD1 = ["interested"]
WORD2 = ["surprised"]

GLOVE_SIZE = 25
glove_file = '../../lib/glove/glove' + str(GLOVE_SIZE) + '.pkl'
glove_embeddings = pickle.load(open(glove_file, 'rb'))


def get_context_score(context, context_embedding):
    for word in context:
        if word in glove_embeddings.keys():
            word_vec = np.array(glove_embeddings[word])
            context_embedding += word_vec
    print(word, context_embedding)
    return context_embedding


def get_sim_scores(score1, score2):
    return (1 - spatial.distance.cosine(score1, score2))


def main():

    # Test values returned for zero-based vs. one-based arrays
    context1_embedding_zeros = np.zeros(GLOVE_SIZE)
    context1_embedding_ones = np.ones(GLOVE_SIZE)

    context2_embedding_zeros = np.zeros(GLOVE_SIZE)
    context2_embedding_ones = np.ones(GLOVE_SIZE)

    score1_zeros = get_context_score(WORD1, context1_embedding_zeros)
    score1_ones = get_context_score(WORD1, context1_embedding_ones)

    score2_zeros = get_context_score(WORD2, context2_embedding_zeros)
    score2_ones = get_context_score(WORD2, context2_embedding_ones)

    print("Zero-based array score", get_sim_scores(score1_zeros, score2_zeros))
    print("One-based array score", get_sim_scores(score1_ones, score2_ones))


if __name__ == '__main__':
    main()
