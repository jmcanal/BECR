"""
Baseline rule-based system for emotion cause extraction of tweets
"""

import sys
from nltk import pos_tag
import numpy as np
np.set_printoptions(threshold=sys.maxsize)
import pickle


# For loading NRC emotion lexicon
emo_matrix = np.load('../lib/emotion_lexicon/NRC/nrc_emotion_lexicon_matrix.npy')
w2idx = pickle.load(open('../lib/emotion_lexicon/NRC/nrc_word_map.pkl', "rb"))
emo2idx = pickle.load(open('../lib/emotion_lexicon/NRC/nrc_emotion_map.pkl', "rb"))
idx2emo = {v: k for k, v in emo2idx.items()}

# For loading curated emotion keyword list
emo_kws = pickle.load(open('../lib/emotion_lexicon/emotion_kw_list/emotion_keywords.pkl', "rb"))


class TweetPatterns():
    # Processes OpenIE outputs
    # - Converts triples to format that can be matched to predefined emotion-cause patterns
    # - Includes preprocessing steps of adding POS tags and emotion values

    def __init__(self, file):
        self.file = file
    def get_patterns(self):
        patterns = []
        tweet_index = {}
        index = 0
        with open(self.file, 'r') as f:
            for line in f:
                if not line.startswith(('0', '1', '\n')):
                    index += 1
                    tweet_index[index] = line
                elif line.startswith('0'):
                    curr_words = line.split(" ", 1)[1]
                    if not curr_words.startswith('Context'):
                        curr_words = curr_words.strip('()\n')
                        curr_words = curr_words.split(';')
                        curr_words = [w for w in curr_words if w is not ""]
                        # print(curr_words)
                        patterns.append((index, pos_tag(curr_words)))
        return patterns, tweet_index


class Rules():
    rule_list = []

    def add_rule(self, new_rule):
        self.rule_list.append(new_rule)

    def apply_rules(self, tweet_triple, tweet_idx):
        # These rules are set up for NLTK POS tag matching
        idx, pattern = tweet_triple
        raw_tweet = tweet_idx[idx]
        first_person = ('I', 'i')
        nominals = ('JJ', 'NN', 'PRP')
        verbs = ('VB', 'MD')
        make = ('makes', 'made', 'has made', 'have made', 'will make')
        modals = ('may', 'might', 'could', 'should', 'would', 'will')
        openie_markup = ('L:', 'T:')
        prep_conj = ('in', 'about', 'for', 'because', 'from', 'at', 'to')
        # also do negated modals version?

        a = pattern[0][0]
        b = pattern[0][1]
        c = pattern[1][0].lstrip()
        d = pattern[1][1]
        e = pattern[2][0].lstrip()
        f = pattern[2][1]

        # Example: "I love Bernie Sanders"
        if a in first_person and d.startswith(verbs) and f.startswith(nominals):
            emo_word = self.check_emos(c)[0]
            if emo_word and e is not "":
                print("1:", raw_tweet, pattern)
                print("emotion:", emo_word, "-->  cause:", e, "\n")

        # Example: "Seeing videos of them performing at digi has made
        # me excited to see the jacks in November"
        elif b.startswith(nominals) and c.startswith(make):
            emo_word = self.check_emos(e)[0]
            if emo_word and not emo_word.startswith('sense'):
                print("2:", raw_tweet, pattern)
                print("emotion:", emo_word, "-->  cause:", a, "\n")

        # Example: "Here's a crazy car fact that might surprise you: Volkswagen owns Bentley"
        # In the future try to catch both parts: 1. "crazy car fact" AND 2. "Volkswagen owns Bentley"
        # another example: "The results may surprise you."
        elif c.startswith(modals) and not e.startswith(openie_markup) and f.startswith(nominals):
            emo_word = self.check_emos(c)[0]
            if emo_word and a not in first_person:
                print("3:", raw_tweet, pattern)
                print("emotion:", emo_word, "-->  cause:", a, "\n")

        # Example: "I'm so excited for the new episode of Hannibal tomorrow *cries*"
        elif f.startswith(verbs):
            emo_word, words, i = self.check_emos(e)
            if emo_word and len(words) > i+2:
                if words[i+1] in prep_conj:
                    print("4:", raw_tweet, pattern)
                    print("emotion:", emo_word, "-->  cause:", " ".join(words[i+2:]), "\n")



    def check_emos(self, phrase):
        words = phrase.split()
        for i, word in enumerate(words):

            # Below is code for NRC emo list
            # if word in w2idx.keys():
            #     idx = w2idx[word]
            #     if np.sum(emo_matrix[idx]) > 0:
            #         return word, words, i

            if word in emo_kws:
                return word, words, i
        return None, None, None


def main():
    """
    Apply rules to tweets and return emotion-cause relations when found for tweets
    :return:
    """

    tweet_file = sys.argv[1]
    patterns, tweet_idx = TweetPatterns(tweet_file).get_patterns()
    for p in patterns:
        Rules().apply_rules(p, tweet_idx)


if __name__ == "__main__":
    main()