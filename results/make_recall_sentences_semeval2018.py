

file = '../results/recall_semeval2018_raw'
output = '../results/run_recall_semeval2018.txt'

with open(file, "r") as f:
    with open(output, "w") as out:
        for line in f:
            if not line.startswith(('0', '1', '\n')):
                out.write(line.rstrip() + "\n")

