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
from src.SnowBreds.seed import Seed
from src.baseline.dependency_rule_extractor import EmotionCauseRuleExtractor
from src.baseline.dependency_tweet_loader import TweetLoader


class RuleBootstrapper:

    # Loading pre-identified emo-cause seed pairs in dictionary format
    seed_pairs = pickle.load(open('../lib/seeds/train_seeds.pkl', "rb"))

    def __init__(self, tau=0.85, cycles=10):
        """
        Initialize by setting tau and cycle thresholds for cosine similarity scores between seed matches and
        candidate seeds
        :param tau: tau threshold value
        :param cycles: cycle value
        """
        self.tau = tau
        self.cycles = cycles

    def get_seed_matches(self, emo_list, tweet_objects):
        """
        set up list of seeds to search for in twitter preprocessed outputs
        :param emo_list: list of emotion cause pairs
        :param tweet_objects: list of Tweet objects from file
        :return: list of Seed objects
        """
        seed_matches = []
        for ex in emo_list:
            emo, cause = ex
            cause_rawtext = " ".join([w.text for w in cause])
            if emo.text in self.seed_pairs.keys():
                if cause_rawtext in self.seed_pairs[emo.text]:
                    new_seed = Seed(emo, cause, cause_rawtext, tweet_objects[emo.tweet_idx])
                    seed_matches.append(new_seed)
                    self.get_seed_contexts(new_seed, emo, cause)
                    emo.seed = True # emo-word is added to seed matches
                    new_seed.cosine = 1.0 # initial seed matches given highest cosine sim score by default
                    new_seed.cycle = 0

        return seed_matches

    def get_seed_contexts(self, seed, emo, cause):
        """
        Assign before, between and after contexts to seed
        :param seed: Seed object
        :param emo: Word object
        :param cause: list of Word objects
        :return:
        """
        if emo.idx < cause[0].idx:
            reln1 = [emo]   # TODO: fix this when switching to multi-word emo
            reln2 = cause
        else:
            reln1 = cause
            reln2 = [emo]

        seed.get_context_before(reln1)
        seed.get_context_btwn(reln1, reln2)
        seed.get_context_after(reln2)

    def cosine_sim(self, seed_match, candidate_seed, alpha=1/3, beta=1/3, gamma=1/3):
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
        sim = before_sim + between_sim + after_sim
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
            if candidate_seed.emo.seed: # If this seed is already a match, skip it
                continue
            else:
                max_cosine = 0

                for seed in seed_matches:
                    cos_sim = self.cosine_sim(seed, candidate_seed, 0.2, 0.6, 0.2)  # todo: make a hyperparameter

                    if cos_sim > tau:
                        max_cosine = cos_sim if cos_sim > max_cosine else max_cosine

                if max_cosine > 0:
                    new_seeds.append(candidate_seed)
                    candidate_seed.emo.seed = True
                    candidate_seed.cosine = max_cosine
                    candidate_seed.cycle = cycle

        seed_matches.extend(new_seeds)

    def set_all_contexts(self, emo_list, tweet_objects):
        """
        Initialize all seed contexts - before, between and after
        :param emo_list: list of Word objects set True for emotion
        :param tweet_objects: Tweet objects
        :return: list of candidate Seed objects
        """
        candidate_seeds = []
        for idx, ex in enumerate(emo_list):
            emo = ex[0]
            if emo.seed:
                continue
            else:
                cause = ex[1]
                cause_rawtext = " ".join([w.text for w in cause])
                candidate_seed = Seed(emo, cause, cause_rawtext, tweet_objects[emo.tweet_idx])
                self.get_seed_contexts(candidate_seed, emo, cause)
                emo.seed = False # initially set to False
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
        for i in range(self.cycles):
            self.find_new_relations(candidates, seed_matches, self.tau, i+1)

        return seed_matches

    def print_emo_causes(self, seed_matches, output):
        """
        Write out the extracted emotion cause relations
        :param seed_matches: list of Seed object matches
        :param output: the output file name
        :return: void
        """
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

    tweets = TweetLoader(parsed_tweet_file)
    tweets.extract_emo_relations()
    extractor = EmotionCauseRuleExtractor()
    emo_list = extractor.build_emo_cause_list(tweets.tweet2emo, tweets.idx2tweet)

    bootstrapper = RuleBootstrapper()
    seed_matches = bootstrapper.get_seed_matches(emo_list, tweets.tweet_list)
    seed_matches = bootstrapper.run_bootstrapping(emo_list, seed_matches, tweets.tweet_list)
    bootstrapper.print_emo_causes(seed_matches, output)


if __name__ == "__main__":
    main()