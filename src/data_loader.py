import os 
import argparse
from xml.etree import ElementTree as ET


DATA_TYPES = (
    "doc_spec",  # Document set specification files 
    "human_summaries", # Human-created gold-standard model summary files
    "baseline_summaries", # Official submission + baseline summary files
)


def get_loader(data_type):
    """
    Args:
        data_type (str): must be in DATA_TYPES and in config.json
    
    Returns:
        The loading function for that data type.
    """
    if data_type == "doc_spec":
        return load_doc_spec
    elif data_type == "human_summaries":
        return load_human_summaries
    elif data_type == "baseline_summaries":
        return load_baseline_summaries
    raise Exception("Data type {} is invalid".format(data_type))


def filenames(data_type, split, year, data_store):
    """
    Args:
        data_type (str): must be in DATA_TYPES and in config.json
        split (str): training, devtest, or evaltest
        year (str): 2009 or 2010 
        data_store (dict): loaded config.json
    
    Returns:
        A generator that gives one filename at a time.
    """
    dirname = os.path.join(data_store[data_type], split, year)   # Base path is the one from the config file
    for f in os.listdir(dirname):
        if data_type == "doc_spec" and not f.endswith(".xml"):
            # Document specification files must have "*.xml" extension, so skip
            continue 
        
        yield os.path.join(dirname, f)


def load_data(data_type, split, year=2010):
    """
    Args:
        data_type (str): must be in DATA_TYPES and in config.json
        split (str): training, devtest, or evaltest
        year (int): data from the year of the task
    """
    assert data_type in data_types 
    
    with open("config.json") as infile:
        data_store = json.load(infile)
    
    fn_generator = filenames(data_type, split, year, data_store)
    loader = get_loader(data_type)
    return loader(fn_generator, data_store)


def load_doc_spec(fn_generator, data_store):
    for f in fn_generator:
        # TODO: Use the document ids in the document set specification 
        # file to retrieve the relevant documents in ACQUAINT & ACQUAINT-2
        pass
    return


def load_human_summaries(fn_generator, data_store):
    for f in fn_generator:
        # TODO
        pass
    return


def load_baseline_summaries(fn_generator, data_store):
    for f in fn_generator:
        # TODO
        pass
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.json")
    parser.add_argument("--split", type=str, default="training", choices=["devtest", "evaltest", "training"])