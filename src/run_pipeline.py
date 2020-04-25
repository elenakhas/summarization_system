import os 
import argparse
import json

from data_loader import load_data
from content_selection import select_content


def run(args):
    if args.run_id is None:
        args.run_id = "D2run0"
        # args.run_id = args.deliverable + datetime.now().strftime('%Y%m%d%H%M%S')

    with open(args.config) as infile:
        data_store = json.load(infile)

    if not os.path.exists(data_store["working_dir"]):
        os.makedirs(data_store["working_dir"])

    input_data = load_data("input_data", data_store, "devtest", year=2010)
    preprocessed_data = preprocess(input_data)
    content = select_content(input_data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.json")
    parser.add_argument("--split", type=str, default="training", choices=["devtest", "evaltest", "training"])
    parser.add_argument("--run_id", default=None)
    args = parser.parse_args() 

    run(args)


    