import os
import json
import spacy
import pandas as pd
from spacy.lang.en import English
import en_core_web_sm
from sklearn.feature_extraction.text import TfidfVectorizer 
import tqdm 

nlp = spacy.load("en_core_web_sm") # package
spacy_stopwords = spacy.lang.en.stop_words.STOP_WORDS

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
#     if want to remove questions, add to the list: if str(sent)[-1] != "?"
    sents = [sent for sent in spacydoc.sents]
    return sents

def get_sents_noquest(spacydoc):
    '''Returns a list of sentences'''
#     if want to remove questions, add to the list: if str(sent)[-1] != "?"
    sents = [sent for sent in spacydoc.sents if str(sent)[-1] != "?"]
    return sents

def tokens_per_sentence(spacysent):
    '''Returns all lowercased tokens'''
    return [token.text.lower() for token in spacysent]

def get_namedentities(spacysentdoc):
    ''' Extracts named entities of place, person, date types
    Inputs: spaCy annotated document
    Outputs: a dictionary witk keys = NE types, values = corresponding tokens
    '''
    namedentities = dict()
    for ent in spacysentdoc.ents:
        if ent.label_ in namedentities:
            namedentities[ent.label_].append(ent.text)
        else:
            namedentities[ent.label_] = [ent.text]
    return namedentities
        
def get_concreteness(spacysentdoc, df):
    '''given a list of strings (tokens of a sentence), returns the concreteness
    score for the sentence. The score is the sum of the score for tokens that can be 
    found in the http://crr.ugent.be/archives/1330 dataset. 
    Inputs: spacysentdoc
    Outputs: a tuple containing the function name and the result. result is an integer rounded to 3 dp 
    and converted to a string
    ''' 
    c_score = 0
    for token in [token.text for token in spacysentdoc]: 
        try:
            c_score += df[df["Word"] == token]["Conc.M"].values[0]
        except:
            pass
    return round(c_score,5)

def get_postags(spacysent):
    '''Performs postagging on a file
    Inputs: spaCy annotated document
    Outputs: a tuple containing the function name and the result (a list of tuples with token)
    
    Uses WordNet postags: https://spacy.io/api/annotation#pos-tagging
    '''
    return [(token.text, token.pos_) for token in spacysent if token.text.isalpha() and not (token.is_punct or token.is_stop)]

def get_parsetags(spacysent):
    parses_sent = []
    for token in spacysent:
        if not token.is_punct:
            parse = (token.text, token.dep_, token.head.text, token.head.pos_)
    #                 [child for child in token.children if not child.is_punct])
            parses_sent.append(parse)
    return parses_sent


def get_noun_chunks(spacysent):
    return [chunk.text.lower() for chunk in spacysent.noun_chunks]

# conversion to spacy doc
def preprocess_data(data):
    documents = {}
    for topic_id in data.keys():
        title = data[topic_id]['title']
        docs = data[topic_id]['docs']
        spacydocs = []
        spacydocs.append(([nlp(title)], 'TITLE'))
        for doc_id, doc in docs.items():
            doc = nlp(doc)
            spacydocs.append((get_sents_noquest(doc), doc_id))
        documents[topic_id] = spacydocs
        
    return documents

def process_document(listwithid, doc_index, topic_id, sentences, concrete_df):
    spacydoc = listwithid[0]
    
    # spacydoc is a list of noninterrogative spacy preprocessed sentences
    total_sents = len(spacydoc) # total number of non-interrogative sentences
    for index, sentence in enumerate(spacydoc):
        sentence_info = {"doc_id": listwithid[1], "doc_index": doc_index, "topic_id": topic_id, "index": index, "lemmas": list(), "length": 0, "total_sent": total_sents,
                       "all_tokens": list(), "postags": list(),"parsetags": list(), "namedent": dict(), "noun_chunks": list(), "concreteness": 0.0 }

        # all tokens
        tokens = tokens_per_sentence(sentence)
        sentence_info["length"] = len(tokens)
        # lemmas for valid words (not stop, only alphabetic)
        sentence_info["lemmas"] = get_lemmas(sentence)
        sentence_info["postags"] = get_postags(sentence)
        sentence_info["parsetags"] = get_parsetags(sentence)
        sentence_info["namedent"] = get_namedentities(sentence)
        sentence_info["concreteness"] = get_concreteness(sentence, concrete_df)
        sentence_info['noun_chunks'] = get_noun_chunks(sentence)
        sentence_info['all_tokens'] = get_tokens(sentence)
        sentences[str(sentence)] = sentence_info
    return sentences


def process_documents_by_topic(documents, concrete_df):
    sentences_info = dict()
    for topic_id in documents.keys():
        sentences_info[topic_id] = dict()
        docs = documents[topic_id]
        for i, doc in tqdm.tqdm(enumerate(docs)):
            process_document(doc, i, topic_id, sentences_info[topic_id], concrete_df)
    return sentences_info


def tfidf_preprocessing(grand_dict):
    all_docs = []
    topic_ids = []
    for topic_id in grand_dict.keys():
        one_topic = []
        topic_ids.append(topic_id)
        
        docs = grand_dict[topic_id]
        for sent, v in docs.items():
            lem = v['lemmas']
            one_topic += lem
        string_doc = " ".join(one_topic)
        all_docs.append(string_doc)
    return all_docs, topic_ids

def get_sentence_tfidfs(lemmas, topic_id, tfidf_df):
    total = 0
    for lemma in lemmas:
        try:
            score = tfidf_df.loc[lemma, topic_id]
        except KeyError:
            score = 0
        total += score    
    return total

def construct_tfidf_dataframe(lemmatized, tfidf_vectorizer):
    preprocessed, ids = tfidf_preprocessing(lemmatized)
    vectors=tfidf_vectorizer.fit_transform(preprocessed)
    dense = vectors.todense()
    tfidf_df = pd.DataFrame(dense, columns=tfidf_vectorizer.get_feature_names(), index=ids).T
    return tfidf_df

def update_with_tfidf(info_by_topic, tfidf_df):
    for topic, info_by_sent in info_by_topic.items():
        for sentence, sentence_info in info_by_sent.items():
            lemmas = sentence_info['lemmas']
            tfidf = get_sentence_tfidfs(lemmas, topic, tfidf_df)
            sentence_info.update({"tf_idf" : tfidf})
        
    return info_by_topic


def preprocess(data, preprocessed_json_path, overwrite=False):
    if os.path.exists(preprocessed_json_path) and not overwrite:
        with open(preprocessed_json_path) as infile:
            return json.load(infile)

    concreteness_file = os.path.join("working_files", "concreteness.txt")
    concrete_df = pd.read_csv(concreteness_file, sep="\t", header=0)

    documents = preprocess_data(data)
    sent_info = process_documents_by_topic(documents, concrete_df)
    tfidf_vectorizer=TfidfVectorizer(use_idf=True)
    tf_idf_df = construct_tfidf_dataframe(sent_info, tfidf_vectorizer)
    updated_dict = update_with_tfidf(sent_info, tf_idf_df)

    with open(preprocessed_json_path, "w") as outfile:
        json.dump(updated_dict, outfile, indent=2)
    return updated_dict