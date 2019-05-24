"""
Baseline rule-based system for emotion cause extraction of tweets
Based on outputs from the TweeboParser Dependency Parser
"""

import sys
# from nltk import pos_tag
import numpy as np
np.set_printoptions(threshold=sys.maxsize)
import pickle
from collections import defaultdict as dd


# For loading NRC emotion lexicon
emo_matrix = np.load('../lib/emotion_lexicon/NRC/nrc_emotion_lexicon_matrix.npy')
w2idx = pickle.load(open('../lib/emotion_lexicon/NRC/nrc_word_map.pkl', "rb"))
emo2idx = pickle.load(open('../lib/emotion_lexicon/NRC/nrc_emotion_map.pkl', "rb"))
idx2emo = {v: k for k, v in emo2idx.items()}

# For loading curated emotion keyword list
emo_kws = pickle.load(open('../lib/emotion_lexicon/emotion_kw_list/emotion_keywords.pkl', "rb"))


class Word:

    def __init__(self, word_feats, tweet):
        # word_feats = input.split('\t')
        self.idx = int(word_feats[0])
        self.text = word_feats[1]
        self.pos = word_feats[3]
        self.parent = int(word_feats[6])
        if word_feats[7] == 'MW' or word_feats[7] == 'CONJ':
            self.mw = word_feats[7]
        else:
            self.mw = None

        self.tweet_idx = tweet
        self.children = []

    def add_child(self, child):
        self.children.append(child)
        # print("TEST", self.idx, child, self.children)

    def has_children(self):
        # print("CHILDREN", self.children)
        return len(self.children) > 0

    def __str__(self):
        return self.text

    def get_children(self):
        return ", ".join([w.text for w in self.children])


class GetTweets:

    # file = sys.argv[1]
    tweet_idx = 0
    # start_idx = 0
    idx2tweet = {}
    tweet_idx_2_emo_words = dd(list)
    idx2word = {}
    child2parent = {}

    def __init__(self, raw_file):
        self.file = raw_file

    def create_words(self):
        full_tweet = []
        prev_word_idx = 0
        VerbAdj = ['V', 'A']
        with open(self.file, 'r') as f:
            for line in f:
                if line is '\n':
                    # update tweet dictionary and add children to words
                    self.add_relatives()
                    self.idx2tweet[self.tweet_idx] = " ".join(full_tweet)
                    # clear and go to next tweet
                    full_tweet = []
                    self.tweet_idx += 1
                    self.idx2word = {}
                    self.child2parent = {}
                    prev_word_idx = 0
                else:
                    line = line.split('\t')
                    word_idx = int(line[0])
                    pos = line[3]
                    if word_idx > prev_word_idx:
                        raw_word = line[1]
                        curr_word = Word(line, self.tweet_idx)
                        self.idx2word[int(word_idx)] = curr_word
                        self.child2parent[curr_word] = int(line[6])
                        if raw_word in emo_kws and pos in VerbAdj:
                            # Isolate emotion words that are Verbs or Adjectives
                            self.tweet_idx_2_emo_words[self.tweet_idx].append(curr_word)
                        full_tweet.append(raw_word)
                        prev_word_idx = word_idx

        return self.tweet_idx_2_emo_words, self.idx2tweet

    def add_relatives(self):
        # print(self.idx2word)
        for child, parent in self.child2parent.items():
            if parent != 0 and parent != -1:
                parent_word = self.idx2word[parent]
                parent_word.add_child(child)
                child.parent = parent_word



def get_dependencies(word_list, deps):
    # Return list of dependencies for a word
    if not word_list:
        return deps
    elif type(word_list) is Word:
        word_list = word_list.children[:]
        return get_dependencies(word_list, deps)
    else:
        word = word_list.pop()
        # print(word.children)
        word_list.extend(word.children)
        deps[word.idx] = word
        return get_dependencies(word_list, deps)

def apply_rules(emo_word):
    deps = []
    MODALS = ('may', 'might', 'could', 'should', 'would', 'will')
    MAKE = ('makes', 'made', 'make')
    TENSE = ('will', 'has', 'had')
    if emo_word.pos == 'V' and emo_word.has_children():
        if emo_word.parent == 0:
            # Apply Rule 1
            # Example: "I love Bernie Sanders"
            deps = sort_dependencies(emo_word, 1)
        elif emo_word.parent.text in MODALS and emo_word.parent.pos == 'V':
            # Apply Rule 2
            # Example: "The results may surprise you."
            deps = sort_dependencies(emo_word.parent, 2)
        elif emo_word.has_children():
            # Apply Rule 3
            # Example: "exhausted from putting groceries away"
            deps = sort_dependencies(emo_word, 3)
    elif emo_word.pos == 'A' and emo_word.has_children():
        # parent = emo_word.parent
        # grandparent = emo_word.parent.parent
        # if (parent.text in MAKE and parent.pos == 'V'):
        #     # Apply Rule 4
        #     # Example: "Bob Marley vs James Brown mashup will make anyone feel good"
        #     deps = sort_dependencies(parent, 4)
        # elif grandparent != 0 and grandparent != -1:
        #     if (emo_word.parent.parent.text in MAKE and emo_word.parent.parent.pos == 'V'):
        #         # Apply Rule 4
        #         # Example: "Bob Marley vs James Brown mashup will make anyone feel good"
        #         if grandparent.parent in TENSE:
        #             deps = sort_dependencies(grandparent.parent, 4)
        #         else:
        #             deps = sort_dependencies(grandparent, 4)

        # Apply Rule 5
        # Example: "I'm so excited for the new episode of Hannibal tomorrow"
        deps = sort_dependencies(emo_word, 5)
    else:
        return emo_word, None

    if deps:
        return emo_word, " ".join(deps)
    else:
        return emo_word, None

def sort_dependencies(word, rule):
    deps = sorted(get_dependencies(word, {}).items())
    LHS = [d.text for (i, d) in deps if i < word.idx]
    RHS = [d.text for (i, d) in deps if i > word.idx]

    if rule == 1 or rule == 3 or rule == 5:
        return RHS
    elif rule == 2 or rule == 4:
        return LHS
    else:
        return 'invalid rule'


def main():
    """
    Apply rules to Tweeboparser outputs and return emotion-cause relations when found for tweets
    :return: void
    """

    parsed_tweet_file = sys.argv[1]
    # output = sys.argv[3]
    emo_words, idx2tweets = GetTweets(parsed_tweet_file).create_words()

    for sent_id, words in emo_words.items():
        print("\nsentence_{}".format(sent_id), idx2tweets[sent_id])
        for word in words:
            emo, cause = apply_rules(word)
            if cause:
                print("EMOTION:", emo, "CAUSE", cause)
            elif emo:
                print("EMOTION:", emo, "NO CAUSE FOUND")
            else:
                print("NO EMOTION FOUND / NO CAUSE FOUND")

    # tp = TweetPatterns(raw_tweet_file, parsed_tweet_file)
    # rules = EmotionCauseRuleExtractor(tp.tweets)
    # for p in tp.patterns:
    #     emotion_cause = rules.apply_rules(p)
    #     if emotion_cause:
    #         tweet_emo_cause.append(emotion_cause)
    # 
    # with open(output, "w") as out:
    #     for line in tweet_emo_cause:
    #         print(line[0], file=out)
    #         print(line[1], file=out)


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
            if word in emo_kws:
                return word, words, i
        return False, words, -1

if __name__ == "__main__":
    main()