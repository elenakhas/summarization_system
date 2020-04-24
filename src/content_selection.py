import argparse 
import os
import json
from data_loader import load_data 
from nltk.tokenize import word_tokenize, sent_tokenize
import string 
from tqdm import tqdm


def create_placeholder_summaries(output_dir, topic_id, doc_id, doc, unique_alphanum="D2run0", subtopic_id=None):
    sentences = sent_tokenize(doc)
    summary_tokens = []
    summary_lines = []
    for sentence_id, sent in enumerate(sentences):
        tokens = word_tokenize(sent)

        if len(tokens) + len(summary_tokens) > 100:
            break
        else:
            summary_tokens.extend(tokens)
            summary_lines.append(sent.strip())
    
    id_part_1 = topic_id[:-1]
    id_part_2 = topic_id[-1]
    output_name = "{id_part_1}-A.M.100.{id_part_2}.{unique_alphanum}".format(
        id_part_1=id_part_1,
        id_part_2=id_part_2,
        unique_alphanum=unique_alphanum,
    )
    output_path = os.path.join(output_dir, output_name)

    print(output_path)

    with open(output_path, "w") as outfile:
        for line in summary_lines:
            outfile.write(line + "\n")

        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.json")
    parser.add_argument("--split", type=str, default="training", choices=["devtest", "evaltest", "training"])
    args = parser.parse_args() 

    with open(args.config) as infile:
        data_store = json.load(infile)

    if not os.path.exists(data_store["working_dir"]):
        os.makedirs(data_store["working_dir"])

    input_data = load_data("input_data", data_store, "training", year=2009)
    
    # topic_id = "D0943H"
    for topic_id in tqdm(input_data.keys()):
        for doc_id, doc in input_data[topic_id]["docs"].items():
            # print("doc_id: {}\ndoc: {}".format(doc_id, doc))
            create_placeholder_summaries(data_store["output_dir"], topic_id, doc_id, doc)
            break