import os 
import argparse
import json
from datetime import datetime
import time

from data_loader import load_data
from content_selection.preprocessing import preprocess 
from content_selection.LDA import sentence_selection_wrapper
from generate_eval_config import write_eval_config
from generate_summaries import make_summaries
from get_embeddings import make_embeddings


def run_module(desc, func, *args, **kwargs):
    """
    Args:
        desc (str): Description of the run
        func (func): Function to run
    All the arguments and keyword arguments are passed to the function.
    """
    print(desc)
    start = time.time()
    return_val = func(*args, **kwargs)
    print("\t finished {} in {}".format(desc, time.time()-start))
    return return_val


def run(args):
    print("running with args: {}".format(args))
    if args.run_id is None:
        args.run_id = args.deliverable + datetime.now().strftime('%Y%m%d%H%M%S')

    with open(args.config) as infile:
        data_store = json.load(infile)

    if not os.path.exists(data_store["working_dir"]):
        os.makedirs(data_store["working_dir"])

    input_data, xml_filename = run_module("loading input data", load_data, "input_data", data_store, 
        args.split, test=args.test, overwrite=False)

    preprocessed_data = run_module(
        "loading preprocessed data", 
        preprocess, 
        input_data, 
        os.path.join(data_store["working_dir"], os.path.basename(xml_filename)[:-4] + ".json.preprocessed"),
        overwrite=False)


    topic_sentences = run_module(
        "selecting content",
        sentence_selection_wrapper,
        preprocessed_data,
        os.path.join(
            data_store["working_dir"], 
            os.path.basename(xml_filename)[:-4] + ".json.selected"),
        num_sentences=args.num_sentences,
        overwrite=False,
    )

    bert_embeddings = run_module(
        "getting sentence embeddings",
        make_embeddings,
        topic_sentences=topic_sentences,
        pickle_path=os.path.join(data_store["working_dir"], 
            "{}_{}_{}.pickle".format(args.model_name, args.deliverable, args.split)),
        model_name=args.model_name,
        overwrite=False,
    )

    run_module(
        "generating sentences",
        make_summaries,
        topic_sentences, bert_embeddings, args, data_store,
        use_embeddings=args.use_embeddings,
        sim_threshold=args.sim_threshold,
        num_sentences=args.num_sentences,
    )

    run_module(
        "writing eval config",
        write_eval_config,
        args, data_store, overwrite=True,
    )
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.json")
    parser.add_argument("--deliverable", type=str, default="D4", help='deliverable number')
    parser.add_argument("--split", type=str, default="training", choices=["devtest", "evaltest", "training"])
    parser.add_argument("--run_id", default=None)
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--use_embeddings", action="store_true")
    parser.add_argument("--model_name", default="bert-base-cased")
    parser.add_argument("--sim_threshold", type=float, default=0.95)
    parser.add_argument("--num_sentences", type=int, default=20)
    args = parser.parse_args()
    run(args)


    
