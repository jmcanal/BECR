"""
Semi-supervised approach to emo-cause extraction based on BREDS model created by David Batista
(https://github.com/davidsbatista/BREDS) which is itself an extension of Snowball
++++++++++++++++++++++++++++
Uses baseline rule-based emo-cause extractor as starting point, also with outputs from the TweeboParser Dependency
Parser
"""
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from scipy import spatial
import pickle
from src.BECR.seed import Seed
from src.BECR.becr_dependency_rule_extractor import BECREmotionCauseRuleExtractor
from src.BECR.becr_dependency_tweet_loader import BECRTweetLoader
import argparse


class RuleBootstrapper:

    def __init__(self, args):
        """
        Initialize by setting tau and cycle thresholds for cosine similarity scores between seed matches and
        candidate seeds
        :param args: args
        """
        self.tau = args.tau
        self.neg_tau = args.neg_tau
        self.cycles = args.cycles
        self.alpha = args.alpha
        self.beta = args.beta
        self.gamma = args.gamma
        self.delta = args.delta
        self.epsilon = args.epsilon
        self.neg_alpha = args.neg_alpha
        self.neg_beta = args.neg_beta
        self.neg_gamma = args.neg_gamma
        self.neg_delta = args.neg_delta
        self.neg_epsilon = args.neg_epsilon
        self.glove_size = args.glove_size
        self.is_test = args.test
        self.predicted_relations = []
        self.seed_pairs = pickle.load(open('../../lib/seeds/train_seeds.pkl', "rb"))

    def get_seed_matches(self, emo_list, tweet_objects):
        """
        Set up list of seeds to search for in twitter preprocessed outputs;
        Seeds must be exact string matches for emotion and cause pair relations
        :param emo_list: list of emotion cause pairs
        :param tweet_objects: list of Tweet objects from file
        :return: list of Seed objects
        """

        # negative seed pairs can be added here
        neg_seed_pairs = {}

        seed_matches = []
        for ex in emo_list:

            emo, cause, sentence = ex
            cause_rawtext = " ".join([w.text for w in cause])
            emo_rawtext = " ".join([w.text for w in emo.phrase])

            if emo_rawtext in neg_seed_pairs.keys():
                if cause_rawtext in neg_seed_pairs[emo_rawtext]:
                    emo.bad_seed = True
                    new_seed = Seed(emo, cause, tweet_objects[emo.tweet_idx], self.glove_size)
                    seed_matches.append(new_seed)
                    self.get_seed_contexts(new_seed, emo, cause)
                    new_seed.cosine = 0  # initial bad seed matches given cosine sim score of 0 by default
                    new_seed.cycle = 0
                    new_seed.bad = True

            if emo_rawtext in self.seed_pairs.keys():
                if cause_rawtext in self.seed_pairs[emo_rawtext]:
                    emo.seed = True # emo-word is added to seed matches
                    new_seed = Seed(emo, cause, tweet_objects[emo.tweet_idx], self.glove_size)
                    seed_matches.append(new_seed)
                    self.get_seed_contexts(new_seed, emo, cause)
                    new_seed.cosine = 1.0 # initial seed matches given highest cosine sim score by default
                    new_seed.cycle = 0

        return seed_matches

    def get_seed_contexts(self, seed, emo, cause):
        """
        Assign before, between and after contexts to seed
        Also get emo and cause embedding scores here
        :param seed: Seed object
        :param emo: Word object
        :param cause: list of Word objects
        :return: void
        """
        if emo.idx < cause[0].idx:
            reln1 = emo.phrase
            reln2 = cause
        else:
            reln1 = cause
            reln2 = emo.phrase

        seed.get_context_before(reln1)
        seed.get_context_btwn(reln1, reln2)
        seed.get_context_after(reln2)

        # Get embedddings for emo and cause
        seed.emo_embedding = seed.calc_glove_score([e.text for e in emo.phrase])
        seed.cause_embedding = seed.calc_glove_score([c.text for c in cause])

    def cosine_sim(self, seed_match, candidate_seed, alpha, beta, gamma, delta, epsilon):
        """
        Calculate cosine similarity for potential seed object with established seed object
        :param seed_match: Seed object, Previously verified seed example
        :param candidate_seed: Seed object; Potential seed match
        :return: similarity score
        """
        # Determine context similarities (spans before, between and after emotion and cause)
        before_sim = alpha * (1 - spatial.distance.cosine(seed_match.bef, candidate_seed.bef))
        between_sim = beta * (1 - spatial.distance.cosine(seed_match.btwn, candidate_seed.btwn))
        after_sim = gamma * (1 - spatial.distance.cosine(seed_match.aft, candidate_seed.aft))

        # Determine Emotion and Cause similarities
        emo_sim = epsilon * (1 - spatial.distance.cosine(seed_match.emo_embedding, candidate_seed.emo_embedding))
        cause_sim = delta * (1 - spatial.distance.cosine(seed_match.cause_embedding, candidate_seed.cause_embedding))

        sim = before_sim + between_sim + after_sim + emo_sim + cause_sim
        return sim

    def find_new_relations(self, candidate_seeds, seed_matches, tau, cycle):
        """
        Find new relations from candidate seeds
        :param candidate_seeds: list of candidate Seed objects
        :param seed_matches: list of matching Seed objects
        :param tau: tau value
        :param cycle: cycle value
        :return: void
        """
        new_seeds = []
        for candidate_seed in candidate_seeds:

            # If this seed is already a match or a bad seed, skip it
            if candidate_seed.emotion.seed or candidate_seed.emotion.bad_seed:
                continue

            max_cosine = 0

            for seed in seed_matches:
                if candidate_seed.bad:
                    continue

                if seed.bad:
                    cos_sim = self.cosine_sim(seed, candidate_seed, self.neg_alpha, self.neg_beta, self.neg_gamma, self.neg_delta, self.neg_epsilon)
                    if cos_sim > self.neg_tau:
                        candidate_seed.bad = True
                else:
                    cos_sim = self.cosine_sim(seed, candidate_seed, self.alpha, self.beta, self.gamma, self.delta, self.epsilon)
                    if cos_sim > tau:
                        max_cosine = cos_sim if cos_sim > max_cosine else max_cosine

            if max_cosine > 0 and not candidate_seed.bad:
                new_seeds.append(candidate_seed)
                candidate_seed.emotion.seed = True
                candidate_seed.cosine = max_cosine
                candidate_seed.cycle = cycle

            elif candidate_seed.bad:
                new_seeds.append(candidate_seed)
                candidate_seed.emotion.bad_seed = True
                candidate_seed.cosine = 0
                candidate_seed.cycle = cycle

        if self.is_test:
            self.predicted_relations.extend(new_seeds)

        else:
            seed_matches.extend(new_seeds)

    def set_all_contexts(self, emo_list, tweet_objects):
        """
        Initialize all seed contexts - before, between and after
        :param emo_list: list of Word objects set True for emotion
        :param tweet_objects: Tweet objects
        :return: list of candidate Seed objects
        """
        candidate_seeds = []
        for example in emo_list:
            emo = example[0]
            if emo.seed or emo.bad_seed:
                continue

            cause = example[1]
            candidate_seed = Seed(emo, cause, tweet_objects[emo.tweet_idx], self.glove_size)
            self.get_seed_contexts(candidate_seed, emo, cause)
            emo.seed = False  # initialize to False
            candidate_seeds.append(candidate_seed)

        return candidate_seeds

    def run_bootstrapping(self, emo_list, seed_matches, tweet_objects):
        """
        Run bootstrapping to get new seed matches
        :param emo_list: list of emotion, cause pairs
        :param seed_matches: list of Seed object matches
        :param tweet_objects: list of Tweet objects
        :return: updated list of Seed object matches
        """
        candidates = self.set_all_contexts(emo_list, tweet_objects)

        # run one cycle in test phase
        if self.is_test:
            self.find_new_relations(candidates, seed_matches, self.tau, 0)
            return self.predicted_relations

        for i in range(self.cycles):
            self.find_new_relations(candidates, seed_matches, self.tau, i + 1)
        return seed_matches

    def print_emo_causes(self, seed_matches, output):
        """
        Write out the extracted emotion cause relations
        :param seed_matches: list of Seed object matches
        :param output: the output file name
        :return: void
        """
        if not self.is_test:
            pickle.dump(seed_matches, open('../../lib/seeds/test_seeds.pkl', 'wb'))

        with open(output, 'w') as out:
            for seed in sorted(seed_matches, key=lambda x: -x.cosine):
                emo_text = " ".join([s.text for s in seed.emotion.phrase])
                cause_text = " ".join([d.text for d in seed.cause])
                relation = "EMOTION: " + emo_text + "\tCAUSE: " + cause_text + "\tTWEET:" + seed.tweet.raw
                print(str(seed.cosine) + " " + relation, file=out)
                print("", file=out)


def parse_args(args):
    """
    Parse arguments from the command line
    :param args: arguments
    :return: list of arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('parsed_tweet_file')
    parser.add_argument('output_file')
    parser.add_argument('--glove_size', choices=[25, 50, 100], default=25)
    # confidence threshold
    parser.add_argument('--tau', type=float, default=0.8)
    # confidence threshold for negative seeds
    parser.add_argument('--neg_tau', type=float, default=0.9)
    # number of iterations
    parser.add_argument('--cycles', type=float, default=10)
    # before context weight
    parser.add_argument('--alpha', type=float, default=0.2)
    # between context weight
    parser.add_argument('--beta', type=float, default=0.5)
    # after context weight
    parser.add_argument('--gamma', type=float, default=0.2)
    # cause weight
    parser.add_argument('--delta', type=float, default=0)
    # emotion weight
    parser.add_argument('--epsilon', type=float, default=0.1)
    # same as alpha - epsilon but for _bad_ seeds
    parser.add_argument('--neg_alpha', type=float, default=0)
    parser.add_argument('--neg_beta', type=float, default=0)
    parser.add_argument('--neg_gamma', type=float, default=0)
    # For bad seeds, the cause is often the best clue, e.g. pronouns "I" and "me"
    parser.add_argument('--neg_delta', type=float, default=0.5)
    # For bad seeds, the verb itself is often the best information
    parser.add_argument('--neg_epsilon', type=float, default=0.5)
    parser.add_argument('--test', action='store_true')
    return parser.parse_args(args)


def main():
    """
    Run BECR algorithm on Tweets and create list of
    emotion-cause relations that are found
    :return: void
    """
    args = parse_args(sys.argv[1:])

    tweets = BECRTweetLoader(args.parsed_tweet_file)
    tweets.extract_emo_relations()
    extractor = BECREmotionCauseRuleExtractor()
    emo_list = extractor.build_emo_cause_list(tweets.tweet2emo, tweets.idx2tweet)

    Seed.load_glove_embeddings(args.glove_size)
    bootstrapper = RuleBootstrapper(args)

    if not args.test:
        seed_matches = bootstrapper.get_seed_matches(emo_list, tweets.tweet_list)
    else:
        # this pickle file is a list of Seed objects
        seed_matches = pickle.load(open('../../lib/seeds/test_seeds.pkl', "rb"))

    seed_matches = bootstrapper.run_bootstrapping(emo_list, seed_matches, tweets.tweet_list)
    bootstrapper.print_emo_causes(seed_matches, args.output_file)


if __name__ == "__main__":
    main()
