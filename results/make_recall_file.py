"""
Randomly sample 20 tweets from for recall calculation
"""
import random
import sys


def create_recall_file(input, output):
    with open(input, "r") as f:
        tweets = f.read().split('\n')
    tweets = [t for t in tweets if not ""]
    sample_tweets = random.sample(tweets, 25)

    with open(output, "w") as out:
        for tweet in sample_tweets:
            print(tweet.split('\n')[0], file=out)


def main():
    """
    Randomly sample 25 tweets for recall file
    :return: void
    """
    input = sys.argv[1]
    output = sys.argv[2]

    create_recall_file(input, output)


if __name__ == "__main__":
    main()

