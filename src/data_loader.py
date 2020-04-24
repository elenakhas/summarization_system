import os 
import argparse
from bs4 import BeautifulSoup
import json
from collections import defaultdict


DATA_TYPES = (
    "input_data",  
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
    if data_type == "input_data":
        return load_input_data
    elif data_type == "human_summaries":
        return load_human_summaries
    elif data_type == "baseline_summaries":
        return load_baseline_summaries
    raise Exception("Data type {} is invalid".format(data_type))


def read_data(corpus_dir, xml_filename, data_store):
    """
    Args:
        corpus_dir (str): path to the corpus as stated in config.json
        xml_filename (str): TAC documents specification
        data_store (dict): loaded config.json
    Returns:
        {"topic_id": {
            "title": title, 
            "narrative": narrative, 
            "docs": {"doc_id": text ... "doc_id": text}}}
    """
    json_path = os.path.join(data_store["working_dir"], os.path.basename(xml_filename)[:-4] + ".json")
    if os.path.exists(json_path):
        with open(json_path) as infile:
            return json.load(infile)

    with open(xml_filename, 'r') as myfile:
        myfile_data = myfile.read().replace('\n', ' ')

    soup = BeautifulSoup(myfile_data, 'lxml')

    names = soup.find_all('topic')
    titles = [element.contents[0].replace("\t", '').strip() for element in soup.find_all("title")]
    narrative = [element.contents[0].replace("\t", '').strip() for element in soup.find_all("narrative")]

    data = {}
    for i in range(len(names)):
        name = names[i]
        topic_id = name.get("id")
        if topic_id not in data:
            data[topic_id] = dict()
            data[topic_id]["title"] = titles[i]
            data[topic_id]["narrative"] = narrative[i].replace("\t", '')
            data[topic_id]["docs"] = {}
            docs = soup.find(id=topic_id + "-A").children
            for doc in docs:
                if str(type(doc)) == "<class 'bs4.element.Tag'>":
                    doc_id = doc["id"]
                    print("fetching {}".format(doc_id))
                    publication = (doc_id.split("_")[0] + "_" + doc_id.split("_")[1]).lower()
                    date = doc_id.split("_")[2].split(".")[0]
                    publication_doc = doc_id.split("_")[2].split(".")[1]

                    path = corpus_dir + publication + "/" + publication + "_" + date[:-2] + ".xml"

                    myfile = open(path, 'r')
                    myfile_data = myfile.read().replace("\n", ' ').strip()
                    document = ""
                    doc_soup = BeautifulSoup(myfile_data, 'lxml')
                    headline = doc_soup.find(id=str(doc_id)).find("headline").contents[0].strip()
                    if doc_soup.find(id=str(doc_id)).find("dateline"):
                        dateline = doc_soup.find(id=str(doc_id)).find("dateline").contents[0].strip()
                    text = doc_soup.find(id=str(doc_id)).find("text").find_all("p")
                    document += headline + ". " + dateline + ". "
                    for line in text:
                        document += line.contents[0].strip() + " "

                    data[topic_id]["docs"][doc_id] = document
    
    print("finished fetching all the data")
    if not os.path.exists(json_path):
        with open(json_path, 'w+') as json_file:
            json.dump(data, json_file)
    return data 


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
    dirname = os.path.join(data_store[data_type], split, str(year))   # Base path is the one from the config file
    for f in os.listdir(dirname):
        if data_type == "input_data" and not f.endswith(".xml"):
            # Document specification files must have "*.xml" extension, so skip
            continue 
        
        yield os.path.join(dirname, f)


def load_data(data_type, data_store, split, year=2010):
    """
    Args:
        data_type (str): must be in DATA_TYPES and in config.json
        split (str): training, devtest, or evaltest
        year (int): data from the year of the task
    """

    assert data_type in DATA_TYPES 
    fn_generator = filenames(data_type, split, year, data_store)
    loader = get_loader(data_type)
    return loader(fn_generator, data_store)


def load_input_data(fn_generator, data_store):
    data = {}
    for f in fn_generator:
        file_data = read_data(data_store["acquaint-2"], f, data_store)
        data.update(file_data)
    return data


def load_human_summaries(fn_generator, data_store):
    for f in fn_generator:
        # TODO
        pass
    raise NotImplementedError


def load_baseline_summaries(fn_generator, data_store):
    for f in fn_generator:
        # TODO
        pass
    raise NotImplementedError



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
