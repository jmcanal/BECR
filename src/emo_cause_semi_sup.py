"""
Semi-supervised approach to emo-cause extraction based on BREDS model
created by David Batista (https://github.com/davidsbatista/BREDS)
which is itself an extension of Snowball
++++++++++++++++++++++++++++
Uses baseline rule-based emo-cause extractor as starting point,
also with outputs from the TweeboParser Dependency Parser
"""


import sys
import numpy as np
from scipy import spatial

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

# Loading glove vectors
GLOVE_SIZE = 25
glove_file = '../lib/glove/glove' + str(GLOVE_SIZE) + '.pkl'
glove_embeddings = pickle.load(open(glove_file, 'rb'))

# Tau threshold for cosine similarity scores between seed matches and candidate seeds
TAU = 0.85
CYCLES = 5


class Word:

    def __init__(self, word_feats, tweet):
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
        self.emo = False
        self.seed = None

    def add_child(self, child):
        self.children.append(child)

    def has_children(self):
        return len(self.children) > 0

    def get_children(self):
        return ", ".join([w.text for w in self.children])

    def has_seed(self):
        """
        True means the Word object has been added to Seed relations
        False means that Word was checked but did not pass threshold
        None means that the word has not been checked for Seed relation
        :return:
        """
        return self.seed

    def __lt__(self, other):
        """
        Words are ordered by their alphanumeric order
        :param other:
        :return:
        """
        return self.text < other.text

    def __str__(self):
        return self.text


class Tweet:

    def __init__(self, raw):
        self.raw = raw
        self.words = []
        self.mapping = {}
        self.emo_words = []
        self.index = None
        self.seed = None

    def __str__(self):
        return self.raw


class GetTweets:

    tweet_idx = 0
    idx2tweet = {}
    tweet_idx_2_emo_words = dd(list)
    idx2word = {}
    child2parent = {}
    tweet_mapping = {}
    tweet_objects = {}

    def __init__(self, raw_file):
        self.file = raw_file

    def create_words(self):
        full_tweet = []
        prev_word_idx = 0
        VERB_ADJ = ['V', 'A']
        with open(self.file, 'r') as f:
            for line in f:
                if line is '\n':
                    # update tweet dictionary & Tweet object and add children to words
                    self.add_relatives()
                    tweet_text = " ".join(full_tweet)
                    self.idx2tweet[self.tweet_idx] = tweet_text

                    # Create Tweet object
                    self.add_tweet(tweet_text, full_tweet)

                    # re-set numbering and go to next tweet
                    full_tweet = []
                    prev_word_idx = 0
                    self.go_to_next_tweet()

                else:
                    line = line.split('\t')
                    word_idx = int(line[0])
                    pos = line[3]
                    if word_idx > prev_word_idx:
                        raw_word = line[1].lower()
                        curr_word = Word(line, self.tweet_idx)
                        self.idx2word[int(word_idx)] = curr_word
                        self.child2parent[curr_word] = int(line[6])
                        if raw_word in emo_kws and pos in VERB_ADJ:
                            # Isolate emotion words that are Verbs or Adjectives
                            self.tweet_idx_2_emo_words[self.tweet_idx].append(curr_word)
                            curr_word.emo = True
                        full_tweet.append(raw_word)
                        prev_word_idx = word_idx
                    self.tweet_mapping[word_idx] = (curr_word, pos)

        return self.tweet_idx_2_emo_words, self.idx2tweet, self.tweet_objects

    def add_tweet(self, tweet_text, tweet_list):
        """
        Create Tweet object
        :param tweet_text: raw text of Tweet
        :param tweet_list: tokenized list of words in Tweet
        :return:
        """
        this_tweet = Tweet(tweet_text)
        this_tweet.words = tweet_list
        this_tweet.index = self.tweet_idx
        this_tweet.mapping = self.tweet_mapping
        this_tweet.emo_words = self.tweet_idx_2_emo_words[self.tweet_idx]
        self.tweet_objects[self.tweet_idx] = this_tweet

    def go_to_next_tweet(self):
        """
        Reset tweet-level data structures and move to next tweet
        :return:
        """
        self.tweet_idx += 1
        self.idx2word = {}
        self.child2parent = {}
        self.tweet_mapping = {}

    def add_relatives(self):
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
        word_list.extend(word.children)
        deps[word.idx] = word
        return get_dependencies(word_list, deps)


def apply_rules(emo_word):
    # Function to grab as examples of emotions and potential causes
    # Acts as pre-processing step to create a set of emo-cause pairs
    # to search first for seed examples
    # and then to find extensions
    deps = []
    MODALS = ('may', 'might', 'could', 'should', 'would', 'will')
    MAKE = ('makes', 'made', 'make')
    TENSE = ('will', 'has', 'had')
    VERBS = ('V', 'L')
    if emo_word.pos in VERBS and emo_word.has_children():
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
    elif emo_word.pos == 'A':
        if emo_word.has_children():
            # Apply Rule 4
            # Example: "I'm so excited for the new episode of Hannibal tomorrow"
            deps = sort_dependencies(emo_word, 4)
        elif emo_word.parent != 0:
            # Apply Rule 5
            # Example: "You may be interested in this evening's BBC documentary"
            if emo_word.parent.pos in VERBS:
                deps = sort_dependencies(emo_word.parent, 5)
    else:
        return emo_word, None

    if deps:
        return emo_word, deps
    else:
        return emo_word, None


def sort_dependencies(word, rule):
    # Extract left-hand and right-hand dependencies of emo-word
    # or emo-word's parent, depending on construction
    # return dependency appropriate to rule definition

    deps = sorted(get_dependencies(word, {}).items())
    LHS_pre = [d for (i, d) in deps if i < word.idx]
    RHS_pre = [d for (i, d) in deps if i > word.idx]

    if rule == 1 or rule == 3 or rule == 4:
        RHS = strip_prepositions(RHS_pre) if RHS_pre else None
        return RHS
    elif rule == 5:
        RHS = strip_prepositions(RHS_pre[1:]) if len(RHS_pre) > 2 else None
        return RHS
    elif rule == 2:
        LHS = strip_prepositions(LHS_pre) if LHS_pre else None
        return LHS
    else:
        return 'invalid rule'


def strip_prepositions(phrase):
    # Strip prepositions, conjunctions from dependent phrase
    PREP_CONJ = ['P', 'R', 'T']

    if phrase:
        if phrase[0].pos in PREP_CONJ:
            return strip_prepositions(phrase[1:])
        else:
            return phrase
    else:
        return None


def preprocess_emo_cause(emo_words):
    # Get the list of examples to search for seed and other emo-cause relations

    emo_lookup = []

    for sent_id, words in emo_words.items():
        for word in words:
            emo, cause = apply_rules(word)
            if cause:
                emo_lookup.append([emo, cause, sent_id])
    return emo_lookup


class Seed:

    def __init__(self, emo, cause, cause_raw, tweet):
        self.emo = emo
        self.cause = cause
        self.cause_raw = cause_raw
        self.tweet = tweet
        self.bef = []
        self.btwn = []
        self.aft = []
        self.cosine = None
        self.cycle = None

    def calc_glove_score(self, context):
        # Calculate context score with GLoVe embedding
        # Will be a vector
        context_embedding = np.ones(GLOVE_SIZE)
        for word in context:    # todo: fix the tokenization; glove has: 's, 'm; twokenizer has i'm, it's
            if word in glove_embeddings.keys():
                word_vec = np.array(glove_embeddings[word])
                context_embedding += word_vec

        return context_embedding

    def get_context_before(self, reln1):
        before = self.tweet.words[0:reln1.idx-1]
        if before:
            self.bef = self.calc_glove_score(before)
        else:
            self.bef = np.ones(GLOVE_SIZE)

    def get_context_btwn(self, reln1, reln2):
        between = self.tweet.words[reln1.idx:reln2[0].idx-1]
        if between:
            self.btwn = self.calc_glove_score(between)
            # print(between, self.btwn)
        else:
            self.btwn = np.ones(GLOVE_SIZE)

    def get_context_after(self, reln2):
        after = self.tweet.words[reln2[-1].idx:-1]
        if after:
            self.aft = self.calc_glove_score(after)
        else:
            self.aft = np.ones(GLOVE_SIZE)


def get_seed_matches(emo_words, idx2tweets, tweet_objects):
    """
    set up list of seeds to search for in twitter preprocessed outputs
    :param emo_words: list of emotion words in Tweet file
    :param idx2tweets: index:tweet lookup dictionary
    :param tweet_objects: list of Tweet objects from file
    :return:
    """
    # todo: move this dictionary to top or somewhere else
    # seed_pairs = {'love': ['life', 'this time of year', 'today', 'christmas time'],
    #               'hate': ['school'],
    #               'excited': ['tomorrow', 'see messi tomorrow'],
    #               'tired': ['waiting'],
    #               'worry': ['the future'],
    #               'afraid': ['tomorrow']}

    seed_pairs = {'love': ['life', 'this time of year', 'christmas time'],
                  'hate': ['school'],
                  'excited': ['tomorrow', 'see messi tomorrow'],
                  'tired': ['waiting'],
                  'worry': ['the future']}

    emo_list = preprocess_emo_cause(emo_words)

    seed_matches = []

    for ex in emo_list:
        emo = ex[0]
        cause = ex[1]
        cause_rawtext = " ".join([w.text for w in cause])
        if emo.text in seed_pairs.keys():
            if cause_rawtext in seed_pairs[emo.text]:
                new_seed = Seed(emo, cause, cause_rawtext,
                                tweet_objects[emo.tweet_idx])
                seed_matches.append(new_seed)
                get_seed_contexts(new_seed, emo, cause)
                emo.seed = True # emo-word is added to seed matches
                new_seed.cosine = 1.0 # initial seed matches given highest cosine sim score by default
                new_seed.cycle = 0

    return emo_list, seed_matches


def get_seed_contexts(seed, emo, cause):
    """
    Assign before, between and after contexts to seed
    :param seed: Seed pbject
    :param emo: Word object
    :param cause: list of Word objects
    :return:
    """
    seed.get_context_before(emo)
    seed.get_context_btwn(emo, cause)
    seed.get_context_after(cause)


def cosine_sim(seed_match, candidate_seed, alpha=1/3, beta=1/3, gamma=1/3):
    """
    Calculate cosine similarity for potential seed object with established seed object
    :param seed_match: Seed object, Previously verified seed example
    :param candidate_seed: Seed object; Potential seed match
    :param alpha: optional weight parameter for before context
    :param beta: optional weight parameter for between context
    :param gamma: optional weight parameter for after context
    :return: float sim - cosine similarity score
    """
    before_sim = alpha * (1 - spatial.distance.cosine(seed_match.bef, candidate_seed.bef))
    between_sim = beta * (1 - spatial.distance.cosine(seed_match.btwn, candidate_seed.btwn))
    after_sim = gamma * (1 - spatial.distance.cosine(seed_match.aft, candidate_seed.aft))
    # print(seed_match.tweet, list(seed_match.bef), candidate_seed.tweet, list(candidate_seed.bef))
    sim = before_sim + between_sim + after_sim
    # print(before_sim, between_sim, after_sim, sim)
    return sim


def find_new_relations(emo_list, seed_matches, tweet_objects, tau, cycle):

    new_seeds = []

    for idx, ex in enumerate(emo_list[2000:2100]):
        emo = ex[0]
        if emo.seed:
            continue
        else: # todo: separate this out; run this part only on first cycle
            cause = ex[1]
            cause_rawtext = " ".join([w.text for w in cause])
            candidate_seed = Seed(emo, cause, cause_rawtext, tweet_objects[emo.tweet_idx])
            get_seed_contexts(candidate_seed, emo, cause)
            emo.seed = False # initially set to False

            for seed in seed_matches:
                max_cosine = 0
                cos_sim = cosine_sim(seed, candidate_seed, 0.2, 0.6, 0.2)

                if cos_sim > tau:
                    max_cosine = cos_sim if cos_sim > max_cosine else max_cosine

            if max_cosine > 0:
                new_seeds.append(candidate_seed)
                candidate_seed.emo.seed = True
                candidate_seed.cosine = max_cosine
                candidate_seed.cycle = cycle

    seed_matches.extend(new_seeds)


def run_bootstrapping(emo_list, seed_matches, tweet_objects):

    tau = TAU
    for i in range(CYCLES):
        find_new_relations(emo_list, seed_matches, tweet_objects, tau, i+1)
        tau = tau + 0 # Perhaps increase tau threshold for each cycle

    return seed_matches

def emo_cause_outputs(seed_matches, output):

    with open(output, 'w') as out:
        for seed in seed_matches:
            out.write(seed.tweet.raw + "\n")
            out.write("EMO: " + seed.emo.text + " CAUSE: " + seed.cause_raw + "\n")
            out.write(str(seed.cosine) + " cycle: " + str(seed.cycle) + "\n")
            out.write("\n")


def main():
    """
    Run modified Snowball / BREDS algorithm on Tweets and create list of
    emotion-cause relations that are found
    :return: void
    """
    parsed_tweet_file = sys.argv[1]
    output = sys.argv[2]

    emo_words, idx2tweets, tweet_objects = GetTweets(parsed_tweet_file).create_words()
    emo_list, seed_matches = get_seed_matches(emo_words, idx2tweets, tweet_objects)
    seed_matches = run_bootstrapping(emo_list, seed_matches, tweet_objects)
    emo_cause_outputs(seed_matches, output)

if __name__ == "__main__":
    main()