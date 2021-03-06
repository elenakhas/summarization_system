import argparse 
import os
import json
from data_loader import load_data 
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk import pos_tag
from tqdm import tqdm
import random
import pprint
from datetime import datetime


def make_placeholder_summaries(input_data, unique_alphanum, out_dir):
    for topic_id in tqdm(input_data.keys()):
        for doc_id, doc in input_data[topic_id]["docs"].items():
            _create_placeholders(topic_id, doc_id, doc, unique_alphanum, out_dir=out_dir)
            break


def make_placeholder_output(input_data, unique_alphanum, out_dir):
    """
    Returns:
        A list of dicts, where each dict represents a sentence. 
        The sentence dict will have the following format:
        {
            "words": list of tokens,
            "pos_tags": list of POS tags,
            "topic_id": topic_id,
            "doc_id": doc_id,
            "subtopic_id": subtopic_id,   # Randomly assigned for now
        }
    """
    all_data = []
    for topic_id in tqdm(input_data.keys()):
        for doc_id, doc in input_data[topic_id]["docs"].items():
            doc_data = _create_placeholders(
                topic_id, doc_id, doc, unique_alphanum)
            all_data.extend(doc_data)
    return all_data


def _create_placeholders(topic_id, doc_id, doc, unique_alphanum, out_dir=None, overwrite=True):
    random.seed(100)
    n_subtopics = 5 

    sentences = sent_tokenize(doc)
    sentences = sentences[2:] # Skip the topic and date line
    summary_tokens = []
    summary_lines = []
    sentence_data = []
    for sentence_id, sent in enumerate(sentences):
        sentence_dict = {}
        subtopic_id = random.randint(0, n_subtopics)
        tokens = word_tokenize(sent)
        pos_tags = [el[1] for el in pos_tag(tokens)]

        if len(tokens) + len(summary_tokens) > 100:
            break
        else:
            summary_tokens.extend(tokens)
            summary_lines.append(sent.strip())

        sentence_dict["words"] = tokens 
        sentence_dict["pos_tags"] = pos_tags
        sentence_dict["topic_id"] = topic_id 
        sentence_dict["doc_id"] = doc_id
        sentence_dict["subtopic_id"] = subtopic_id 
        sentence_data.append(sentence_dict)

    if out_dir is not None:
        id_part_1 = topic_id[:-1]
        id_part_2 = topic_id[-1]
        output_name = "{id_part_1}-A.M.100.{id_part_2}.{unique_alphanum}".format(
            id_part_1=id_part_1,
            id_part_2=id_part_2,
            unique_alphanum=unique_alphanum,
        )
        output_path = os.path.join(out_dir, output_name)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        if not os.path.exists(output_path) or overwrite:
            with open(output_path, "w") as outfile:
                for line in summary_lines:
                    outfile.write(line + "\n")
    
    return sentence_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.json")
    parser.add_argument("--split", type=str, default="training", choices=["devtest", "evaltest", "training"])
    parser.add_argument("--deliverable", type=str, default="D2")
    args = parser.parse_args() 

    with open(args.config) as infile:
        data_store = json.load(infile)

    if not os.path.exists(data_store["working_dir"]):
        os.makedirs(data_store["working_dir"])

    input_data = load_data("input_data", data_store, "devtest", year=2010, test=False)

    # run_id = args.deliverable + datetime.now().strftime('%Y%m%d%H%M%S')
    run_id = "D2run0"
    make_placeholder_summaries(input_data, run_id, data_store["devtest_outdir"])
    placeholder_output = make_placeholder_output(
        input_data, run_id, data_store["devtest_outdir"])
    for sent in placeholder_output[:3]:
        pprint.pprint(sent)
