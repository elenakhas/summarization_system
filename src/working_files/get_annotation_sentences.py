import json
import csv
import sys
import re

def read(infile: str):
    open('annotate_data.csv', 'w').close()
    with open(infile) as f:
        data = json.load(f)

        for i in data:
            if int(re.findall(r'\d+', i)[0]) > 922:
                with open('annotate_data.csv', mode='a') as csv_file:
                    data_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                    for k in data[i].keys():
                        data_writer.writerow([i, k])


if __name__ == "__main__":
    read(sys.argv[1])