import argparse
import operator
import os
from nltk.tokenize import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer
from nltk import pos_tag
from itertools import permutations
from compute_similarity import remove_stopwords, add_lemmas
import spacy
spacy_lm = spacy.load("en_core_web_lg")
import re


def strip_attribution(line, n=4):
    attribution_pattern = re.compile('[,]([^,\'\"]*?)[.]$')
    match = attribution_pattern.search(line)
    attribution_words = ("said", "stated", "according")
    if match is not None and any(w in match.group(1) for w in attribution_words):
        if len(word_tokenize(match.group(1))) <= n:
            # print(match.group(1))
            line = re.sub(attribution_pattern, ".", line)
    return line

def make_summaries(topic_dict, args, data_store):
    """
    given a topic dictionary, generates summaries for each topic
    version with info ordering
    Args:
        topic_dict: dictionary of {topic_id: {sentence: {sentence info}}}

    Returns: dictionary of summaries for each topic id. key: topic_id,
    value: list of summary sentences

    """

    summary_dict = dict()
    for topic_id in topic_dict.keys():
        #print(topic_id)
        summary = []
        summ_length = 0
        sorted_keys = sorted(topic_dict[topic_id], key=lambda x: (topic_dict[topic_id][x]['LDAscore']), reverse=True)
        #print(sorted_keys)


        for sentence in sorted_keys:
            if summ_length >= 100:
                break

            # ignore short sentences
            sen_length = topic_dict[topic_id][sentence]['length']
            if sen_length <= 8 or sen_length > 50:
                continue

            sentence = apply_heuristics_to_sentence(sentence)

            tokens = apply_heuristics_to_tokens(word_tokenize(sentence))


            if summ_length + len(tokens) <= 100:
                summ_length += len(tokens)

                summary.append(TreebankWordDetokenizer().detokenize(tokens))
            else:
                continue # keep going in case we find a shorter sentence to add

        # print("length of summary is {}".format(summ_length))

        # do information ordering for summary
        best_summary = score_coherence(summary)
        print("best summary is {}".format(best_summary))

        summary_dict[topic_id] = best_summary
        # print(summary)
    # print("length of summary dict is {}".format(len(summary_dict)))

    if args.split == "training":
        out_dir = data_store["training_outdir"]
    elif args.split == "devtest":
        out_dir = os.path.join(data_store["devtest_outdir"], args.deliverable)
    for topic_id, sentences in summary_dict.items():
        write_to_file(out_dir, args.run_id, topic_id, sentences)

def score_coherence(summary):
    if len(summary) == 1:
        return summary

    perms = permutations(summary, len(summary))
   # spacy_lm = spacy.load("en_core_web_lg")  #TODO we do not want to be loading this here - way too slow
    candidate_dict = dict()

    # go through all the permutations of sentence orderings
    ord_count = 1
    for p in perms:

        for i in range(1, len(p)):
            #print(p[i-1], p[i])

            #calculate_similarity(p[i - 1], p[i])

            s1 = spacy_lm(p[i-1].lower())
            s2 = spacy_lm(p[i].lower())
            s1_no_stop = remove_stopwords(s1, spacy_lm)
            s2_no_stop = remove_stopwords(s2, spacy_lm)
            s1_processed = add_lemmas(s1_no_stop, spacy_lm)
            s2_processed = add_lemmas(s2_no_stop, spacy_lm)

            #print(s1_processed.similarity(s2_processed))
            cos_score = calculate_similarity(s1_processed, s2_processed)

            try:
                candidate_dict[ord_count] += cos_score
            except KeyError:
                candidate_dict[ord_count] = cos_score

            #print(candidate_dict)

        ord_count += 1

    #print(candidate_dict)
    # divide by n-1
    for option in candidate_dict.keys():
        candidate_dict[option] = candidate_dict[option] / (ord_count - 1)



    return max(candidate_dict.items(), key=operator.itemgetter(1))[0]


def calculate_similarity(s1, s2):
    """
    calculates similarity score for a sentence pair
    Args:
        s1: first sentence
        s2: second sentence

    Returns: cosine similarity score

    """
    return s1.similarity(s2)

# def make_summaries(topic_dict, args, data_store):
#     """
#     given a topic dictionary, generates summaries for each topic
#     Args:
#         topic_dict: dictionary of {topic_id: {sentence: {sentence info}}}
#
#     Returns: dictionary of summaries for each topic id. key: topic_id,
#     value: list of summary sentences
#
#     """
#
#     summary_dict = dict()
#     for topic_id in topic_dict.keys():
#         #print(topic_id)
#         summary = []
#         summ_length = 0
#         sorted_keys = sorted(topic_dict[topic_id], key=lambda x: (topic_dict[topic_id][x]['LDAscore']), reverse=True)
#         #print(sorted_keys)
#
#         for sentence in sorted_keys:
#             if summ_length >= 100:
#                 break
#
#             # ignore short sentences
#             sen_length = topic_dict[topic_id][sentence]['length']
#             if sen_length <= 8 or sen_length > 50:
#                 continue
#
#             sentence = apply_heuristics_to_sentence(sentence)
#
#             tokens = apply_heuristics_to_tokens(word_tokenize(sentence))
#
#
#             if summ_length + len(tokens) <= 100:
#                 summ_length += len(tokens)
#
#                 summary.append(TreebankWordDetokenizer().detokenize(tokens))
#             else:
#                 continue # keep going in case we find a shorter sentence to add
#
#         # print("length of summary is {}".format(summ_length))
#         summary_dict[topic_id] = summary
#         # print(summary)
#     # print("length of summary dict is {}".format(len(summary_dict)))
#
#     if args.split == "training":
#         out_dir = data_store["training_outdir"]
#     elif args.split == "devtest":
#         out_dir = os.path.join(data_store["devtest_outdir"], args.deliverable)
#     for topic_id, sentences in summary_dict.items():
#         write_to_file(out_dir, args.run_id, topic_id, sentences)


def apply_heuristics_to_sentence(sentence):
    #print(sentence)
    #sentence = sentence.replace('or so', '')
    # remove parenthetical expressions () []
    sentence = re.sub("[\(\[].*?[\)\]]", " ", sentence)

    # remove expressions between -- --
    regexp = re.compile(r"([\-])\1.*\1\1")
    sentence = re.sub(regexp, " ", sentence)

    # remove unnecessary phrases
    sentence = sentence.replace('As a matter of fact, ', '')
    sentence = sentence.replace('At this point, ', '')
    sentence = sentence.replace(', however,', '')
    sentence = sentence.replace(', also, ', '')


    # remove ages
    sentence = re.sub(", aged \d+,", "", sentence)

    # remove gerunds
    sentence = re.sub(", [a-z]+[ing][\sa-zA-Z\d]+,", "", sentence)
    # (, [a-z]+[ing][\sa-zA-Z\d]+,| ^[A-Za-z]+[ing][\sa-zA-Z\d]+,)
    return sentence.strip()

def apply_heuristics_to_tokens(tokens):
    # get rid of adverbs
    pos_tags = [el[1] for el in pos_tag(tokens)]

    adverb_indices = [i for i in range(len(pos_tags)) if 'RB' in pos_tags[i]]

    # don't get rid of adverb at end of sentence
    if len(pos_tags) - 2 in adverb_indices:
        adverb_indices.remove(len(pos_tags) - 2)


    # for word in tokens:
        #if word == 'so':
           # i = tokens.index('so')
            #print(i)
            #adverb_indices.remove(i)

    for i in sorted(adverb_indices, reverse=True):
        tokens.pop(i)

    # make sure the first letter of the sentence is capitalized
    tokens[0] = tokens[0].capitalize()

    return tokens


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
            line = line.replace("\\", "").replace(" ,", ", ").replace(" .", ". ").replace("_", " ").replace("  ", " ")
            line = re.sub('["] ([A-Za-z0-9])', '"\g<1>', line)  # Replace `" The` with `"The` 
            line = line.replace(', "', '," ')
            line = line.replace("``", ' "')

            line = strip_attribution(line)
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


