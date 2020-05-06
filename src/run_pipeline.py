import os 
import argparse
import json
from datetime import datetime
import time

from data_loader import load_data
from content_selection.preprocessing import preprocess 
from content_selection.lda import lda_analysis
from generate_eval_config import write_eval_config
from generate_summaries import make_summaries


def run(args):
    print("running with args: {}".format(args))
    if args.run_id is None:
        args.run_id = "D2run0"
        # args.run_id = args.deliverable + datetime.now().strftime('%Y%m%d%H%M%S')

    with open(args.config) as infile:
        data_store = json.load(infile)

    if not os.path.exists(data_store["working_dir"]):
        os.makedirs(data_store["working_dir"])

    print("loading input data")
    start = time.time()
    input_data, xml_filename = load_data("input_data", data_store, args.split, test=args.test)
    print("\tfinished loading input data in {}".format(time.time()-start))

    print("loading preprocessed data")
    start = time.time()
    preprocessed_data = preprocess(input_data, os.path.join(
        data_store["working_dir"], os.path.basename(xml_filename)[:-4] + ".json.preprocessed"))
    print("\tfinished preprocessing data in {}".format(time.time()-start))

    print("selecting content")
    start = time.time()
    topic_sentences = lda_analysis(preprocessed_data, os.path.join(
        data_store["working_dir"], os.path.basename(xml_filename)[:-4] + ".json.selected"))
    print("\tfinished selecting content in {}".format(time.time()-start))
    
    print("generating summaries")
    start = time.time()
    make_summaries(topic_sentences, args, data_store)
    print("\tfinished generating summaries in {}".format(time.time()-start))

    print("writing eval config")
    start = time.time()
    write_eval_config(args, data_store, overwrite=True)
    print("\tfinished writing eval config in {}".format(time.time()-start))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.json")
    parser.add_argument("--deliverable", type=str, default="D2", help='deliverable number, i.e. D2')
    parser.add_argument("--split", type=str, default="training", choices=["devtest", "evaltest", "training"])
    parser.add_argument("--run_id", default=None)
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()
    run(args)


    