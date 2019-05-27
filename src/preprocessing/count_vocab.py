import sys
from collections import Counter

def countInFile(filename):
    with open(filename, mode='r', errors='ignore') as f:
        words = f.read().encode('latin-1', 'ignore').decode('utf-8')
    with open(filename, mode='w') as tok:
        print(words, file=tok)
    return Counter(words.split())


def main():
    """
    Create emotion lexicon matricies and lists from lexicon
    :return:
    """
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    counts = countInFile(input_file)
    with open(output_file, 'w') as f:
        print('\n'.join([word + "\t" + str(counts[word]) for word in sorted(counts)]), file=f)


if __name__ == "__main__":
    main()