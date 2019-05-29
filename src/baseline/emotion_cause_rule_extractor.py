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
    
    def __lt__(self, other):
        """
        Words are ordered by their alphanumeric order
        :param other:
        :return:
        """
        return self.text < other.text


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
        return emo_word, " ".join([d.text for d in deps])
    else:
        return emo_word, None

def sort_dependencies(word, rule):
    noun_verb = ['^', 'O', 'D', 'V', '$', 'N']
    prep_conj = ['P', 'R', 'T']

    deps = sorted(get_dependencies(word, {}).items())
    LHS_pre = [d for (i, d) in deps if i < word.idx]
    RHS_pre = [d for (i, d) in deps if i > word.idx]

    LHS = strip_prepositions(LHS_pre) if LHS_pre else None
    RHS = strip_prepositions(RHS_pre) if RHS_pre else None

    if rule == 1 or rule == 3 or rule == 5:
        # if RHS:
        #     print(word, RHS[0].pos, [(w.text, w.pos) for w in RHS],[(w.text, w.pos) for w in RHS_pre])
        return RHS
    elif rule == 2 or rule == 4:
        # if LHS:
        #     print(word, LHS[0].pos, [(w.text, w.pos) for w in LHS], [(w.text, w.pos) for w in LHS_pre])
        return LHS
    else:
        return 'invalid rule'


def strip_prepositions(phrase):
    # Strip prepositions, conjunctions from dependent phrase
    prep_conj = ['P', 'R', 'T']

    if phrase:
        if phrase[0].pos in prep_conj:
            return strip_prepositions(phrase[1:])
        else:
            return phrase
    else:
        return None




def main():
    """
    Apply rules to Tweeboparser outputs and return emotion-cause relations when found for tweets
    :return: void
    """

    parsed_tweet_file = sys.argv[1]
    # output = sys.argv[3]
    emo_words, idx2tweets = GetTweets(parsed_tweet_file).create_words()

    emo_list = []

    for sent_id, words in emo_words.items():
        # print("\nsentence_{}".format(sent_id), idx2tweets[sent_id])
        for word in words:
            emo, cause = apply_rules(word)
            if cause:
                print("EMOTION:", emo.text, "CAUSE", cause)
                emo_list.append((emo.text, cause, idx2tweets[sent_id]))
            elif emo:
                # print("EMOTION:", emo, "NO CAUSE FOUND")
                continue
            else:
                # print("NO EMOTION FOUND / NO CAUSE FOUND")
                continue

    # emo_list = sorted(emo_list)
    # with open(sys.argv[2], 'w') as out:
    #     for pair in emo_list:
    #         out.write("EMOTION: " + pair[0] + " CAUSE: " + pair[1] + " SENTENCE: " + pair[2] + "\n")


if __name__ == "__main__":
    main()