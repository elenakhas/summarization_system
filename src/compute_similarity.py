import spacy
import sys
import csv



def remove_stopwords(input, spacy_lm):
    result = [token.text for token in input if token.text not in spacy_lm.Defaults.stop_words]
    return spacy_lm(" ".join(result))

def add_lemmas(input, spacy_lm):
    result = []
    for token in input:
        if token.is_punct:
            continue
        #if token.lemma_ == '-PRON-'
            #continue
        result.append(token.lemma_)
    #print(result)
    return spacy_lm(" ".join(result))


def compute(csv_input):
    spacy_lm = spacy.load("en_core_web_lg")
    #print(spacy_lm.Defaults.stop_words)
    #t1 = "Peter Jennings, the last of the three long-dominant network anchors still on the job, said Tuesday he has been diagnosed with lung cancer."
    #t2 = "I am going to try a random sentence and see what it makes of it. Hmm I wonder."
    #print(print("test", spacy_lm(t1).similarity(spacy_lm(t2))))
    with open(csv_input) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                #print(f'Column names are {", ".join(row)}')
                line_count += 1
            else:

                s1 = spacy_lm(row[1].lower())
                s2 = spacy_lm(row[2].lower())
                s1_no_stop = remove_stopwords(s1, spacy_lm)
                s2_no_stop = remove_stopwords(s2, spacy_lm)
                s1_processed = add_lemmas(s1_no_stop, spacy_lm)
                s2_processed = add_lemmas(s2_no_stop, spacy_lm)
                print(s1_processed.similarity(s2_processed))
                line_count += 1

        #print(f'Processed {line_count} lines.')

if __name__ == "__main__":
    compute(sys.argv[1])