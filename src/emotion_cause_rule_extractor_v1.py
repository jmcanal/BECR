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


class TweetPatterns:
    """
    Processes OpenIE outputs
    - Converts triples to format that can be matched to predefined emotion-cause patterns
    - Includes preprocessing steps of adding POS tags and emotion values
    """

    patterns = []
    tweets = []

    def __init__(self, relation_file):
        """
        Initialize the class by splitting the OpenIE file into tweet descriptor sections
        :param relation_file: the OpenIE output file with relations extracted from tweets
        """
        with open(relation_file, 'r') as f:
            self.tweet_relations = f.read().split('\n\n')
        self.get_patterns()

    def get_patterns(self):
        """
        Filter the output relations to find patterns we want to include
        :return: the list of patterns and the raw tweets
        """
        for idx, tweet in enumerate(self.tweet_relations):

            tweet_info = tweet.split('\n')
            self.tweets.append(tweet_info[0])

            for relation in tweet_info[1:]:
                triple = relation.split(" ", 1)[1].strip('()\n')

                # context is not a relation that we care about
                if triple.startswith('Context'):
                    continue

                triple = [w for w in triple.split('; ') if w]
                if len(triple) > 2:
                    self.patterns.append((idx, pos_tag(triple)))


class EmotionCauseRuleExtractor:
    """
    Apply rules to extract emotion causes from tweets
    """

    FIRST_PERSON = ('I', 'i')
    NOMINALS = ('JJ', 'NN', 'PRP')
    VERBS = ('VB', 'MD')
    MAKE = ('makes', 'made', 'has made', 'have made', 'will make')
    MODALS = ('may', 'might', 'could', 'should', 'would', 'will')
    OPENIE_MARKUP = ('L:', 'T:')
    PREP_CONJ = ('in', 'about', 'for', 'because', 'from', 'at', 'to')
    # TODO: negated modals

    def __init__(self, tweets):
        """
        Initialize this class by storing tweets
        :param tweets:
        """
        self.tweets = tweets
        self.rule_list = []

    def is_ifeel(self, phrase1, phrase2, tag2, tag3):
        """
        Is this an 'ifeel' causal relation?

        Example: I love Bernie Sanders

        :param phrase1: the first phrase
        :param phrase2: the second phrase
        :param tag2: pos tags for the second phrase
        :param tag3: pos tags for the third phrase
        :return: bool
        """
        right_form = phrase1 in self.FIRST_PERSON \
                     and tag2.startswith(self.VERBS) \
                     and tag3.startswith(self.NOMINALS)

        return right_form and self.get_emotion_word(phrase2)

    def is_itmakes(self, phrase2, phrase3, tag1):
        """
        Is this an 'itmakes' causal relation?

        Example: Seeing videos of them performing at digi has made me excited to see the jacks in November

        :param phrase2: the second phrase
        :param phrase3: the third phrase
        :param tag1: pos tags for the first phrase
        :return: bool
        """
        right_form = tag1.startswith(self.NOMINALS) \
                     and phrase2.startswith(self.MAKE)

        return right_form and not phrase3.startswith('sense')

    def is_modnom(self, phrase1, phrase2, phrase3, tag3):
        """
        Is this a 'modnom' causal relation?

        Example: Here's a crazy car fact that might surprise you: Volkswagen owns Bentley
        Example: The results may surprise you.

        :param phrase1: the first phrase
        :param phrase2: the second phrase
        :param phrase3: the third phrase
        :param tag3: pos tags for the third phrase
        :return: bool
        """
        right_form = phrase2.startswith(self.MODALS) \
                     and not phrase3.startswith(self.OPENIE_MARKUP) \
                     and tag3.startswith(self.NOMINALS)

        return right_form and phrase1 not in self.FIRST_PERSON

    def is_emoverb(self, phrase3, tag3):
        """
        is this an 'emoverb' causal relation?

        Example: I'm so excited for the new episode of Hannibal tomorrow *cries*

        :param phrase3: the third phrase
        :param tag3: pos tags for the third phrase
        :return: bool
        """
        right_form = tag3.startswith(self.VERBS)
        return right_form and self.get_emotion_word(phrase3) and not (set(self.PREP_CONJ) & set(phrase3.split()))

    def apply_ifeel_rule(self, phrase2, phrase3):
        """
        Extract the emotion and cause for 'ifeel' causal relation

        Example: I love Bernie Sanders

        :param phrase2: the second phrase
        :param phrase3: the third phrase
        :return: the emotion word and cause
        """
        emo_word = self.get_emotion_word(phrase2)[0]
        if not emo_word:
            return False, phrase3
        return emo_word, phrase3

    def apply_itmakes_rule(self, phrase1, phrase3):
        """
        Extract the emotion and cause for 'itmakes' causal relation

        Example: Seeing videos of them performing at digi has made me excited to see the jacks in November

        :param phrase1: the first phrase
        :param phrase3: the third phrase
        :return: the emotion word and cause
        """
        emo_word = self.get_emotion_word(phrase3)[0]
        if not emo_word:
            return False, phrase1
        return emo_word, phrase1

    def apply_modnom_rule(self, phrase1, phrase2):
        """
        Extract the emotion and cause for 'modnom' causal relation

        Example: Here's a crazy car fact that might surprise you: Volkswagen owns Bentley
        Example: The results may surprise you.

        :param phrase1: the first phrase
        :param phrase2: the second phrase
        :return: the emotion word and cause
        """
        emo_word = self.get_emotion_word(phrase2)[0]
        if not emo_word:
            return False, phrase1
        return emo_word, phrase1

    def apply_emoverb_rule(self, phrase3):
        """
        Extract the emotion and cause for 'emoverb' causal relation

        Example: I'm so excited for the new episode of Hannibal tomorrow *cries*

        :param phrase3: the third phrase
        :return: the emotion word and cause
        """
        emo_word, words, i = self.get_emotion_word(phrase3)
        if not emo_word or len(words) <= i + 2:
            return False, phrase3
        return emo_word, " ".join(words[i+2:])

    def apply_rules(self, tweet_triple):
        """
        Try to apply the rules to extract emotion and cause from the given relation triple
        :param tweet_triple: the relation triple from OpenIE
        :return: string of emotion and cause
        """
        # These rules are set up for NLTK POS tag matching
        emotion = cause = False
        idx, pattern = tweet_triple

        phrases, tags = zip(*pattern)
        phrase1, phrase2, phrase3, *rest = phrases
        tag1, tag2, tag3, *rest = tags

        if self.is_ifeel(phrase1, phrase2, tag2, tag3):
            emotion, cause = self.apply_ifeel_rule(phrase2, phrase3)

        if not emotion and self.is_itmakes(phrase2, phrase3, tag1):
            emotion, cause = self.apply_itmakes_rule(phrase1, phrase3)

        if not emotion and self.is_modnom(phrase1, phrase2, phrase3, tag3):
            emotion, cause = self.apply_modnom_rule(phrase1, phrase2)

        if not emotion and self.is_emoverb(phrase3, tag3):
            emotion, cause = self.apply_emoverb_rule(phrase3)


        if not emotion or not cause:
            return None

        emo_cause = "emotion: " + emotion + ", cause: " +  cause + "\n"
        return self.tweets[idx], emo_cause


    def get_emotion_word(self, phrase):
        """
        Get the emotion word and index from the given phrase
        :param phrase: the phrase
        :return: emotion word, list of words in the phrase, index of emotion word
        """
        words = phrase.split()
        for i, word in enumerate(words):

            # Below is code for NRC emo list
            # if word in w2idx.keys():
            #     idx = w2idx[word]
            #     if np.sum(emo_matrix[idx]) > 0:
            #         return word, words, i

            if word in emo_kws:
                return word, words, i
        return False, words, -1


def main():
    """
    Apply rules to OpenIE relations and return emotion-cause relations when found for tweets
    :return: void
    """

    tweet_relation_file = sys.argv[1]
    output = sys.argv[2]
    tweet_emo_cause = []

    tp = TweetPatterns(tweet_relation_file)
    rules = EmotionCauseRuleExtractor(tp.tweets)
    for p in tp.patterns:
        emotion_cause = rules.apply_rules(p)
        if emotion_cause:
            tweet_emo_cause.append(emotion_cause)

    with open(output, "w") as out:
        for line in tweet_emo_cause:
            print(line[0], file=out)
            print(line[1], file=out)

if __name__ == "__main__":
    main()