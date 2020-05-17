import os

dirname = "/home2/schan2/573/summarization_system/outputs/D3"

# wc -w is used during grading

for f in os.listdir(dirname):
    with open(os.path.join(dirname, f)) as infile:
        contents = infile.read()
        words = contents.split()  # split on whitespace
        print("{}\t{}".format(f, len(words)))