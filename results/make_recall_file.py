"""
Randomly sample 20 tweets from for recall calculation
"""
import random
import sys


def create_recall_file(input_file, output_file):
    """
    Create the recall file
    :param input_file: the input file
    :param output_file: the output file
    :return: void
    """
    with open(input_file, "r") as f:
        tweets = f.read().split('\n')
    tweets = [t for t in tweets if not ""]
    sample_tweets = random.sample(tweets, 25)

    with open(output_file, "w") as out:
        for tweet in sample_tweets:
            print(tweet.split('\n')[0], file=out)


def main():
    """
    Randomly sample 25 tweets for recall file
    :return: void
    """
    input_file = sys.argv[1]
    output_file = sys.argv[2]

    create_recall_file(input_file, output_file)


if __name__ == "__main__":
    main()

