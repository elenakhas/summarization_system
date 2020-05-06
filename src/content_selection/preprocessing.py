import json
import spacy
from spacy.lang.en import English
from tqdm import tqdm
import os

'''
Utility functions - all performed for a spacy annotated document (sentence by sentence)
'''
def get_tokens(spacysentdoc):

    return [token.text.lower() for token in spacysentdoc if not token.is_punct and not token.text in spacy_stopwords]

def get_lemmas(spacysentdoc, tag = None):
    '''
    Lemmatizes the tokens of a specific POS or all tokens
    Inputs: spaCy annotated document, optional - tag, a specified POS tag
    The default value None would process all the tokens
    Outputs: a list of lemmas for the selected tokens
    '''
    if tag == None:
        lemmas = [token.lemma_.lower() for token in spacysentdoc if token.text.isalpha() and not (token.is_stop or token.is_punct)]
    else:
        lemmas = [token.lemma_.lower() for token in spacysentdoc if token.pos_== tag]
    return lemmas

def get_sents(spacydoc):
    '''Returns a list of sentences'''
    # if want to remove questions, add to the list: if str(sent)[-1] != "?"
    sents = [sent for sent in spacydoc.sents]
    return sents

def get_sents_noquest(spacydoc):
    '''Returns a list of sentences'''
    # if want to remove questions, add to the list: if str(sent)[-1] != "?"
    sents = [sent for sent in spacydoc.sents if str(sent)[-1] != "?"]
    return sents

def tokens_per_sentence(spacysent):
    '''Returns all lowercased tokens'''
    return [token.text.lower() for token in spacysent]

# conversion to spacy doc
def preprocess_data(nlp, data):
    documents = {}
    for topic_id in tqdm(data.keys()):
        docs = data[topic_id]['docs']
        spacydocs = []
        for doc_id, doc in docs.items():
            doc = nlp(doc)
            spacydocs.append((get_sents_noquest(doc), doc_id))
        documents[topic_id] = spacydocs
    return documents

def process_document(listwithid, doc_index, topic_id, sentences):
    spacydoc = listwithid[0]
    # spacydoc is a list of noninterrogative spacy preprocessed sentences
    total_sents = len(spacydoc) # total number of non-interrogative sentences
    for index, sentence in enumerate(spacydoc):
        sentence_info = {"doc_id": listwithid[1], "doc_index": doc_index, "topic_id": topic_id, "index": index, "lemmas": list(), "length": 0, "total_sent": total_sents}
        # all tokens
        tokens = tokens_per_sentence(sentence)
        sentence_info["length"] = len(tokens)
        # lemmas for valid words (not stop, only alphabetic)
        sentence_info["lemmas"] = get_lemmas(sentence)
        sentences[str(sentence)] = sentence_info
    return sentences

def process_documents_by_topic(documents):
    sentences_info = dict()
    for topic_id in documents.keys():
        sentences_info[topic_id] = dict()
        docs = documents[topic_id]
        for i, doc in enumerate(docs):
            process_document(doc, i, topic_id, sentences_info[topic_id])
    return sentences_info

def preprocess(data, preprocessed_json_path):
    if os.path.exists(preprocessed_json_path):
        with open(preprocessed_json_path) as infile:
            return json.load(infile)
    
    # MAIN #
    # data - the input from the previous stage - a json-like dictionary
    nlp = spacy.load("en_core_web_sm") # package
    spacy_stopwords = spacy.lang.en.stop_words.STOP_WORDS #312 stopwords
    documents = preprocess_data(nlp, data) # returns a dict of lemmas; can dump to json, can keep it as a dict; documents - spacy preprocessed docs; dict {topic: list of docs}
    # summarize all the available info; takes a while
    sent_info = process_documents_by_topic(documents) # returns a dict {topic_id : {sentence string : {info dictionary}}}   
    with open(preprocessed_json_path, "w") as outfile:
        json.dump(sent_info, outfile, indent=2)
    return sent_info