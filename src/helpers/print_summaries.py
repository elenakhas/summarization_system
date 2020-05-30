import os

dirname = "/home2/schan2/573/summarization_system/outputs/D4_devtest"

# wc -w is used during grading
filenames = os.listdir(dirname)
filenames.sort()

for f in filenames:
    with open(os.path.join(dirname, f)) as infile:
        topic_id = f.partition("-")[0]
        number = int(topic_id[-2:])
        if number <= 23:
            contents = infile.read()
            print(topic_id)
            print(contents)
        # words = contents.split()  # split on whitespace
        # print("{}\t{}".format(f, len(words)))