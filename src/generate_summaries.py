import argparse
import os
import json
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from nltk.util import ngrams
from nltk.corpus import stopwords 
import re
import string


def overlap_exists(query_toks, sentences, n=4):
    query_ngrams = set(ngrams(query_toks, n))
    for sent_toks in sentences:
        sent_ngrams = set(ngrams(sent_toks, n))
        if query_ngrams & sent_ngrams:
            return True
    return False


def strip_attribution(line, n=8):
    attribution_pattern = re.compile('[,]([^,\'\"]*?)[.]$')
    match = attribution_pattern.search(line)
    attribution_words = ("said", "stated", "according")
    if match is not None and any(w in match.group(1) for w in attribution_words):
        if len(word_tokenize(match.group(1))) <= n:
            print(match.group(1))
            line = re.sub(attribution_pattern, ".", line)
    return line
    

def make_summaries(topic_dict, args, data_store):
    """
    given a topic dictionary, generates summaries for each topic
    Args:
        topic_dict: dictionary of {topic_id: {sentence: {sentence info}}}

    Returns: dictionary of summaries for each topic id. key: topic_id,
    value: list of summary sentences

    """
    summary_dict = dict()
    stop_words = set(stopwords.words('english')) 
    for topic_id in topic_dict.keys():
        #print(topic_id)
        summary = []

        summary_lemmas = []
        summary_tokens = []

        summ_length = 0
        sorted_keys = sorted(topic_dict[topic_id], key=lambda x: (topic_dict[topic_id][x]['LDAscore']), reverse=True)
        #print(sorted_keys)

        for orig_sentence in sorted_keys:
            sentence = orig_sentence
            if summ_length >= 100:
                break

            # remove all sentences that contain quotes
            # quotes = ('"', "'", "`")
            # if any(q in sentence for q in quotes):
            #     continue

            sentence = strip_attribution(sentence)
            # rmeove parenthetical expressions
            sentence = re.sub("[\(\[].*?[\)\]]", "", sentence)
            tokens = word_tokenize(sentence)
            tokens = [tok for tok in tokens if tok != "_"]
            pos_tags = [el[1] for el in pos_tag(tokens)]
            lemmas = topic_dict[topic_id][orig_sentence]["lemmas"]

            if overlap_exists(lemmas, summary_lemmas):
                continue
            
            # ignore short sentences
            if topic_dict[topic_id][orig_sentence]['length'] <= 8:
                continue
           

            adverb_indices = [i for i in range(len(pos_tags)) if pos_tags[i] == 'RB']
            #print(adverb_indices)
            for i in sorted(adverb_indices, reverse=True):
                tokens.pop(i)
            #print(tokens, pos_tags)
            if summ_length + len(tokens) <= 100:
                summ_length += len(tokens)

                summary.append(TreebankWordDetokenizer().detokenize(tokens))
                summary_lemmas.append(lemmas)
                summary_tokens.append(tokens)
            else:
                continue # keep going in case we find a shorter sentence to add

        # print("length of summary is {}".format(summ_length))
        summary_dict[topic_id] = summary
        # print(summary)
    # print("length of summary dict is {}".format(len(summary_dict)))

    if args.split == "training":
        out_dir = data_store["training_outdir"]
    elif args.split == "devtest":
        out_dir = data_store["devtest_outdir"]
    for topic_id, sentences in summary_dict.items():
        write_to_file(out_dir, args.run_id, topic_id, sentences)
    

def write_to_file(out_dir, run_id, topic_id, sentences):
    id_part_1 = topic_id[:-1]
    id_part_2 = topic_id[-1]
    output_name = "{id_part_1}-A.M.100.{id_part_2}.{unique_alphanum}".format(
        id_part_1=id_part_1,
        id_part_2=id_part_2,
        unique_alphanum=run_id,
    )
    output_path = os.path.join(out_dir, output_name)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    with open(output_path, "w") as outfile:
        for line in sentences:
            # Formatting things that have no effect on word count
            line = line.replace("\\", "").replace(" ,", ", ").replace(" .", ". ").replace("_", " ").replace("  ", " ")
            line = re.sub('["] ([A-Za-z0-9])', '"\g<1>', line)  # Replace `" The` with `"The` 
            line = line.replace(', "', '," ')
            line = line.replace("``", ' "')
            outfile.write(line + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("candidates", type=str, default='candidate_sentences.json')
    parser.add_argument("run_id", type=str, default="D2run0")
    parser.add_argument("out_dir", type=str, default="../outputs/D2")
    args = parser.parse_args()
    with open(args.candidates) as infile:
        topic_dictionary = json.load(infile)

    summaries_dict = make_summaries(topic_dictionary)



