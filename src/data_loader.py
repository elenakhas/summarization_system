import os 
import argparse
from bs4 import BeautifulSoup
import json
from collections import defaultdict
from datetime import datetime
from tqdm import tqdm, trange


DATA_TYPES = (
    "input_data",  
    "human_summaries", # Human-created gold-standard model summary files
    "baseline_summaries", # Official submission + baseline summary files
)


def get_path_from_docid(doc_id, split, data_store):
    if "_" in doc_id:  # Then the document is in ACQUAINT-2
        corpus_dir = data_store["acquaint-2"]
        publication = (doc_id.split("_")[0] + "_" + doc_id.split("_")[1]).lower()
        date = doc_id.split("_")[2].split(".")[0]
        publication_doc = doc_id.split("_")[2].split(".")[1]
        path = corpus_dir + publication + "/" + publication + "_" + date[:-2] + ".xml"
        return path
    else:  # Then the document is in ACQUAINT
        corpus_dir = data_store["acquaint"]
        publication = doc_id[0:3]
        date = doc_id.split(".")[0][3:]
        year = date[0:4]
        if publication == "NYT":
            path = corpus_dir + publication.lower() + "/" + year + "/" + date + "_" + publication
        elif publication == "XIE":
            path = corpus_dir + publication.lower() + "/" + year + "/" + date + "_" + "XIN_ENG"
        else:
            path = corpus_dir + publication.lower() + "/" + year + "/" + date + "_" + publication + "_ENG"
        return path


def process_acquaint1(path, doc_id):
    print("processing {} in process_acquaint1".format(path))
    myfile = open(path, 'r')
    myfile_data = myfile.read().replace("\n", ' ').strip()
    document_string = ""
    doc_soup = BeautifulSoup(myfile_data, 'lxml')
    results = doc_soup.find_all("docno")
    for result in results:
        if result.contents[0].strip() == doc_id:
            docno = result
            break
    
    doc = docno.parent

    if doc.find("headline"):
        headline = doc.find("headline").contents[0].strip()
    else:
        headline = ""
    if doc.find("date_time").contents[0].strip():
        datetime = doc.find("date_time").contents[0].strip()
    else:
        datetime = ""

    if doc.find("category"):
        category = doc.find("category").contents[0].strip()
    else:
        category = ""
    text = doc.find("text").find_all("p")
    document_string += headline + ". " + datetime + ". " + category + ". "

    for line in text:
        document_string += line.contents[0].strip() + " "
    return document_string


def process_acquaint2(path, doc_id):
    print("processing {} in process_acquaint2".format(path))
    myfile = open(path, 'r')
    myfile_data = myfile.read().replace("\n", ' ').strip()
    document_string = ""
    doc_soup = BeautifulSoup(myfile_data, 'lxml')

    headline = ""
    dateline = ""
    if doc_soup.find(id=str(doc_id)).find("headline"):
        headline = doc_soup.find(id=str(doc_id)).find("headline").contents[0].strip()
    if doc_soup.find(id=str(doc_id)).find("dateline"):
        dateline = doc_soup.find(id=str(doc_id)).find("dateline").contents[0].strip()
    text = doc_soup.find(id=str(doc_id)).find("text").find_all("p")

    # document_string += headline + ". " + dateline + ". "
    if headline:
        document_string += headline + ". "
    if dateline:
        document_string += dateline + ". "
    for line in text:
        document_string += line.contents[0].strip() + " "
    return document_string    


def read_data(xml_filename, split, data_store, test=False):
    """
    Args:
        xml_filename (str): TAC documents specification
        data_store (dict): loaded config.json
    Returns:
        {"topic_id": {
            "title": title, 
            "narrative": narrative, 
            "docs": {"doc_id": text ... "doc_id": text}}}
    """
    json_path = os.path.join(data_store["working_dir"], os.path.basename(xml_filename)[:-4] + ".json")
    if test:
        json_path += ".small"
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
    for i in trange(len(names), desc="topic"):
        name = names[i]
        topic_id = name.get("id")
        if topic_id not in data:
            data[topic_id] = dict()
            data[topic_id]["title"] = titles[i]

            if split != "devtest":
                data[topic_id]["narrative"] = narrative[i].replace("\t", '')

            data[topic_id]["docs"] = {}
            docs = soup.find(id=topic_id + "-A").children
            for doc in docs:
                if str(type(doc)) == "<class 'bs4.element.Tag'>":
                    doc_id = doc["id"]
                    path = get_path_from_docid(doc_id, split, data_store)
                    if "_" in doc_id: # Then the document is in the newer ACQUAINT-2
                        document = process_acquaint2(path, doc_id)
                    else:  # Then the document is in the older ACQUAINT-1
                        document = process_acquaint1(path, doc_id)

                    data[topic_id]["docs"][doc_id] = document

    
    print("finished fetching all the data")
    if not os.path.exists(json_path):
        with open(json_path, 'w+') as json_file:
            print("writing to {}".format(json_path))
            json.dump(data, json_file, indent=2)
            print("finished writing to {}".format(json_path))
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
    dirname = os.path.join(data_store[data_type], split)   # Base path is the one from the config file
    if split == "training":
        dirname = os.path.join(dirname, str(year))
    for f in os.listdir(dirname):
        if data_type == "input_data" and not f.endswith(".xml"):
            # Document specification files must have "*.xml" extension, so skip
            continue 
        yield os.path.join(dirname, f)


def load_data(data_type, data_store, split, test=False, year=2010):
    """
    Args:
        data_type (str): must be in DATA_TYPES and in config.json
        split (str): training, devtest, or evaltest
        year (int): data from the year of the task
    """

    assert data_type in DATA_TYPES 
    fn_generator = filenames(data_type, split, year, data_store)
    data = {}
    for f in fn_generator:
        file_data = read_data(f, split, data_store, test=test)
        data.update(file_data)
    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.json")
    parser.add_argument("--split", type=str, default="training", choices=["devtest", "evaltest", "training"])
    args = parser.parse_args() 

    with open(args.config) as infile:
        data_store = json.load(infile)

    if not os.path.exists(data_store["working_dir"]):
        os.makedirs(data_store["working_dir"])

    input_data = load_data("input_data", data_store, "devtest", year=2010)
