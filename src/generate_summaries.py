import argparse
import operator
import os
from nltk.tokenize.treebank import TreebankWordDetokenizer
import nltk
from itertools import permutations
from compute_similarity import remove_stopwords, add_lemmas
import re
from scipy.spatial.distance import cosine


CAPS_PATTERN = re.compile("([A-Z]{2,}(?:\s[A-Z-'0-9]{2,})*)")  # matches one or more consecutive capitalized words
ATTR_PATTERN = re.compile('[,]([^,\'\"]*?)[.]$')
PARENS_PATTERN = re.compile("[\(\[].*?[\)\]]")
QUOTESPACE_PATTERN = re.compile('["] ([A-Za-z0-9])')
SENTENCE_VERSIONS = dict() # multiple sentence versions, key: doc_index_index
PRINT_REDUNDANT = True


def strip_attribution(line, n=5):
    match = ATTR_PATTERN.search(line)
    attribution_words = ("said", "say", "report", "state", "according")
    if match is not None and any(w in match.group(1) for w in attribution_words):
        if len(nltk.word_tokenize(match.group(1))) <= n:
            line = ATTR_PATTERN.sub(".", line)
    return line


def make_summaries(topic_dict, embeddings, args, data_store, sim_threshold=0.95, min_length=8, max_length=50, use_embeddings=False):
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
        summary = []
        full_summary = []  # without truncated sentences
        summ_length = 0
        sorted_keys = sorted(topic_dict[topic_id], key=lambda x: (topic_dict[topic_id][x]['LDAscore']), reverse=True)

        for orig_sentence in sorted_keys:
            sentence = orig_sentence

            # ignore sentences containing capitalized words 
            match = CAPS_PATTERN.search(sentence)
            if match is not None:
                continue
             
            if "/" in sentence:
                continue

            if '"' in sentence:
                continue

            # store original sentence version
            sentence_id = "{doc_index}_{index}".format(
                doc_index=topic_dict[topic_id][sentence]['doc_index'],
                index=topic_dict[topic_id][sentence]['index'],
            )
            SENTENCE_VERSIONS[sentence_id] = [sentence]

            # check if sentence is redundant with existing sentences
            if summary:
                redundant = check_sim_threshold(summary, full_summary, sentence, topic_dict[topic_id], 
                    embeddings, sim_threshold=sim_threshold, use_embeddings=use_embeddings)
                if redundant:
                    # TODO: choose the longest sentence version
                    continue


            if summ_length >= 100:
                break

            # ignore short sentences
            sen_length = topic_dict[topic_id][sentence]['length']
            if sen_length <= min_length or sen_length > max_length:
                continue

            sentence = apply_heuristics_to_sentence(sentence)
            
            # Make sure that the sentence starts with an alphanumeric character
            start_index = 0
            for c in sentence:
                if c.isalnum():
                    break
                else:
                    start_index += 1
            sentence = sentence[start_index:]

            tokens = apply_heuristics_to_tokens(nltk.word_tokenize(sentence))


            if summ_length + len(tokens) <= 100:
                summ_length += len(tokens)

                summary.append(TreebankWordDetokenizer().detokenize(tokens))
                full_summary.append(orig_sentence)
            else:
                continue # keep going in case we find a shorter sentence to add
        # do information ordering for summary
        best_summary = score_coherence(summary, full_summary, embeddings=embeddings)
        summary_dict[topic_id] = best_summary
    out_dir = data_store["{}_outdir".format(args.split)]
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    for topic_id, sentences in summary_dict.items():
        write_to_file(out_dir, args.run_id, topic_id, sentences)


def check_sim_threshold(summary, full_summary, sentence, topic_dict, embeddings, sim_threshold=0.95, use_embeddings=False):
    """
    Checks if a sentence is redundant with sentences already in summary.
    if yes, adds it to SENTENCE_VERSIONS
    Args:
        summary: the sentences in the summary so far
        sentence: the sentence being evaluated
        topic_dict: dict with sentence info

    Returns: True if redundant

    """
    for s, orig_s in zip(summary, full_summary):
        if use_embeddings:
            similarity = 1 - cosine(embeddings[orig_s], embeddings[sentence])
        else:
            raise Exception("spaCy similarity is no longer used: run with --use_embeddings flag")
            # similarity = calculate_similarity(s, sentence)
        if similarity > sim_threshold:
            if PRINT_REDUNDANT:
                print("redundant pair {}: \n {} \n {}\n".format(similarity, s, sentence))
            SENTENCE_VERSIONS["{}_{}".format(topic_dict[sentence]['doc_index'],
                                             topic_dict[sentence]['index'])].append(sentence)
            return True

    return False


def score_coherence(summary, full_summary, embeddings):
    if len(summary) == 1:
        return summary

    perms = permutations(range(len(summary)), len(summary))
    candidate_dict = dict()
    # go through all the permutations of sentence orderings
    ord_count = 1
    for p in perms:

        for i in range(1, len(p)):
            cos_score = 1 - cosine(embeddings[full_summary[i-1]], embeddings[full_summary[i]])
            try:
                candidate_dict[p] += cos_score
            except KeyError:
                candidate_dict[p] = cos_score

        ord_count += 1
    # divide by n-1
    for option in candidate_dict.keys():
        candidate_dict[option] = candidate_dict[option] / (ord_count - 1)

    best_ordering = max(candidate_dict.items(), key=operator.itemgetter(1))[0]
    return [summary[i] for i in best_ordering]


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
    days_of_week = "monday tuesday wednesday thursday friday saturday sunday \
        mon. tue. wed. thur. fri. sat. sun.".split()
    # get rid of adverbs
    pos_tags = [el[1] for el in nltk.pos_tag(tokens)]

    remove_indices = list()
    last_index = len(pos_tags)-1
    for i, pos in enumerate(pos_tags):
        if pos == "RB" and tokens[i].lower() != "when":
            if i < last_index and pos_tags[i+1] == "IN":
                continue
            remove_indices.append(i)
        
        if tokens[i].lower() in days_of_week:
            remove_indices.append(i)
            if i != 0 and pos_tags[i-1] == "IN":
                remove_indices.append(i-1)

    
    # don't get rid of adverb at end of sentence
    if len(pos_tags) - 2 in remove_indices:
        remove_indices.remove(len(pos_tags) - 2)

    for i in sorted(remove_indices, reverse=True):
        tokens.pop(i)

    # make sure the first token is alphanumeric
    if not tokens[0].isalnum() and tokens[0] != '"':
        tokens = tokens[1:]

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
            line = QUOTESPACE_PATTERN.sub('"\g<1>', line)  # Replace `" The` with `"The` 
            line = line.replace(', "', '," ')
            line = line.replace("``", ' "')

            line = strip_attribution(line)
            
            if not line[0].isupper():
                line = line[0].upper() + line[1:]
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


