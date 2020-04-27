import argparse
import json 
import os 
from datetime import datetime

from data_loader import load_data 
from content_selection import make_placeholder_output


def order_information(selected_content):
    # TODO
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.json")
    parser.add_argument("--split", type=str, default="training", choices=["devtest", "evaltest", "training"])
    args = parser.parse_args() 

    with open(args.config) as infile:
        data_store = json.load(infile)

    if not os.path.exists(data_store["working_dir"]):
        os.makedirs(data_store["working_dir"])

    input_data = load_data("input_data", data_store, "training", year=2009, test=True)
    placeholder_output = make_placeholder_output(input_data, data_store)
    for sent in placeholder_output[:5]:
        pprint.pprint(sent)