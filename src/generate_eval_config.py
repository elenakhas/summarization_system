import argparse
import xml.etree.ElementTree as ET
from os import listdir
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("deliverable", type=str, default="D2", help='deliverable number, i.e. D2')
    parser.add_argument("data", type=str, default="training", choices=["devtest", "evaltest", "training"], help='type of data, i.e. training')
    parser.add_argument("year", type=str, default="2009", help="release year of data")
    args = parser.parse_args()

    outf = "rouge_run_{}_{}.xml".format(args.deliverable, args.data)
    if args.data == 'training':
        outputs_path = "../outputs/train_output"
        output_files = [f for f in listdir(outputs_path) if isfile(join(outputs_path, f))]
        #model_path = "/Users/esgardner/PycharmProjects/" + args.year # for running locally
        model_path = "/dropbox/19-20/573/Data/models/training/" + args.year # for running on patas
        #model_files = ""
        model_files = [f for f in listdir(model_path) if isfile(join(model_path, f)) and '-A' in f]
    elif args.data == 'devtest':
        outputs_path = "../outputs"
        output_files = [f for f in listdir(outputs_path) if isfile(join(outputs_path, f))]
        model_path = "/dropbox/19-20/573/Data/models/devtest"
        model_files = [f for f in listdir(model_path) if isfile(join(model_path, f)) and '-A' in f]

    #print(output_files)
    #print(sorted(output_files))
    print(len(model_files))
    build_tree(outf, sorted(output_files), outputs_path, sorted(model_files), model_path)

