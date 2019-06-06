"""
Script for converting labeled training data to seed dictionary for BECR algorithm
"""

import sys
import pickle
from collections import defaultdict as dd


class LabelToSeed:

    seeds = dd(list)
    tweets = []
    labels = []

    def find_emo_cause(self, tweet_file, label_file):
        """
        Find the emotion causes
        :param tweet_file: tokenized tweet file
        :param label_file: labeled tweet file
        :return: void
        """
        with open(tweet_file, 'r') as t:
            for words in t:
                words = words.split()
                words = [w.split(":")[0] for w in words]
                self.tweets.append(words)
        with open(label_file, 'r') as l:
            for tags in l:
                tags = tags.split()
                self.labels.append(tags)
        for idx in range(len(self.labels)):
            self.extract_emo_cause(self.labels[idx], self.tweets[idx])

        self.pickle_seeds()

    def extract_emo_cause(self, labels, words):
        """
        Extract the emotions and causes
        :param labels: list of labels
        :param words: list of words
        :return: void
        """
        emo_flag = False
        cause_flag = False
        emo = []
        cause = []

        if len(labels) != len(words):
            raise Warning("Tweet tokens and labels do not match: ", Warning)
        else:
            for idx, label in enumerate(labels):

                if label == "I-E":
                    emo.append(words[idx])

                elif label == "I-C":
                    cause.append(words[idx])

                elif label == "O":
                    if emo_flag:
                        emo_flag = False
                        if cause:
                            self.seeds[" ".join(emo)].append(" ".join(cause))
                            emo = []
                            cause = []
                    elif cause_flag:
                        cause_flag = False
                        if emo:
                            self.seeds[" ".join(emo)].append(" ".join(cause))
                            emo = []
                            cause = []

                elif label == "B-E":
                    if cause_flag:
                        cause_flag = False
                        if emo:
                            self.seeds[" ".join(emo)].append(" ".join(cause))
                            emo = []
                            cause = []
                    emo.append(words[idx])
                    emo_flag = True

                elif label == "B-C":
                    if emo_flag:
                        emo_flag = False
                        if cause:
                            self.seeds[" ".join(emo)].append(" ".join(cause))
                            emo = []
                            cause = []
                    cause.append(words[idx])
                    cause_flag = True

                else:
                    raise Warning("Unknown label encountered: " + label, Warning)

            # When BIO tags occur at end of sentence
            if emo_flag or cause_flag:
                if emo and cause:
                    self.seeds[" ".join(emo)].append(" ".join(cause))

    def pickle_seeds(self):
        """
        Pickle the seed data
        :return: void
        """
        pickle.dump(self.seeds, open('../../lib/seeds/train_seeds.pkl', 'wb'))


def main():
    """
    Convert labeled tweets to training seeds for the BECR algorithm
    :return: void
    """
    tweet_tokens = sys.argv[1]
    tweet_labels = sys.argv[2]

    LabelToSeed().find_emo_cause(tweet_tokens, tweet_labels)


if __name__ == "__main__":
    main()
