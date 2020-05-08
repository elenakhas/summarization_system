import argparse
import xml.etree.ElementTree as ET
import os
from os.path import isfile, join
from typing import List


def indent(elem, level=0):
    """
    breaks up xml tree output so each tag is on an indented line
    """
    i = "\n" + level*"  "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "  "
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
        for elem in elem:
            indent(elem, level+1)
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = i


def build_tree(outfile: str, outputs_dir: List, outputs_path: str, models_dir: List, model_path: str):
    """
    builds an xml tree and writes it to given output file
    Args:
        outfile: file to write to
        outputs_dir: list of files in outputs directory
        outputs_path: path to outputs dir
        models_dir: list of files in models directory
        model_path: path to models dir

    Returns: None

    """
    #tree = ET.ElementTree()
    top = ET.Element('ROUGE_EVAL', version="1.5.5") # make root
    # model file D0901-A.M.100.A.A
    # make eval subtrees
    for f in outputs_dir:
        EVAL_ID = f[:-7]
        #print(EVAL_ID)
        EVAL = ET.SubElement(top, 'EVAL', ID=EVAL_ID)
        PEER_ROOT = ET.SubElement(EVAL, 'PEER-ROOT')
        PEER_ROOT.text = outputs_path
        MODEL_ROOT = ET.SubElement(EVAL, 'MODEL-ROOT')
        MODEL_ROOT.text = model_path
        INPUT_FORMAT = ET.SubElement(EVAL, "INPUT-FORMAT", TYPE="SPL")
        PEERS = ET.SubElement(EVAL, "PEERS")
        P = ET.SubElement(PEERS, "P", ID="1") #TODO: verify that P ID is always 1
        P.text = f
        MODELS = ET.SubElement(EVAL, "MODELS")
        # add all the models
        models = [model for model in models_dir if EVAL_ID in model]
        for m in models:
            M = ET.SubElement(MODELS, "M", ID=m[-1])
            M.text = m

    #ET.dump(top)
    tree = ET.ElementTree(top)
    indent(top)
    tree.write(outfile, encoding="utf-8", short_empty_elements=False)


def write_eval_config(args, data_store, overwrite=True):
    # print("args.split: {}".format(args.split))

    outf = "rouge_run_{}_{}.xml".format(args.deliverable, args.split)
    if args.split == 'training':
        outputs_path = data_store["training_outdir"]
        if outputs_path.endswith("/"):
            outputs_path = outputs_path[:-1]
        output_files = [f for f in os.listdir(outputs_path) if isfile(join(outputs_path, f))]
        #model_path = "/Users/esgardner/PycharmProjects/" + args.year # for running locally
        model_path = os.path.join(data_store["human_summaries"], args.split, "2009")
        model_files = [f for f in os.listdir(model_path) if isfile(join(model_path, f)) and '-A' in f]
    elif args.split == 'devtest':
        outputs_path = os.path.join(data_store["devtest_outdir"], args.deliverable)
        if outputs_path.endswith("/"):
            outputs_path = outputs_path[:-1]
        output_files = [f for f in os.listdir(outputs_path) if isfile(join(outputs_path, f))]
        model_path = os.path.join(data_store["human_summaries"], args.split)
        model_files = [f for f in os.listdir(model_path) if isfile(join(model_path, f)) and '-A' in f]
    build_tree(outf, sorted(output_files), outputs_path, sorted(model_files), model_path)


if __name__ == "__main__":
    import json
    # Testing for module
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.json")
    parser.add_argument("--deliverable", type=str, default="D2", help='deliverable number, i.e. D2')
    parser.add_argument("--split", type=str, default="training", choices=["devtest", "evaltest", "training"])
    parser.add_argument("--run_id", default=None)
    parser.add_argument("--test", default=False)
    args = parser.parse_args()

    with open(args.config) as infile:
        data_store = json.load(infile)

    if not os.path.exists(data_store["working_dir"]):
        os.makedirs(data_store["working_dir"])

    write_eval_config(args, data_store)

