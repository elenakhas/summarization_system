import os
import sys
import json
import gensim
import math
from gensim import corpora
from gensim.utils import simple_preprocess
from gensim.models import LdaModel, LdaMulticore


#reaad json file
def parseJson(json_file):
    '''
    parsing the JSON file from the pre-processing pipeline
    :param json_file
    :return: dictionary with docID as key
    '''

    with open(json_file) as f:
        data = json.load(f)

    #print(data['D0901A']['docs'])
    #print(data.keys())
    return data


def get_document_collection(input_data):
    '''
    this function gets the entire corpus for idf score calculation
    :param data:
    :return: list of documents
    '''
    doc_list = []
    for key, value in input_data.items():
        _doc_list = []
        for k, v in input_data[key].items():
            _doc_list.append(' '.join(input_data[key][k]['lemmas']))

        doc_list.append(_doc_list)

    #print('Number of documents::')
    #print(len(doc_list))
    return doc_list

def get_idf_scores(doc_list):
    '''
    computes IDF score at a document level and not at a corpus level, note that the different documents belong to disjointed themes
    so the IDF scores will keep increasing as we include more documents.
    :param text:
    :return:
    '''
    idf_scores = {}
    num_docs = len(doc_list)
    for doc in doc_list:
        for sen in doc:
            for w in sen.split(' '):
                if w not in idf_scores:
                    idf_scores[w] = 1
                else:
                    idf_scores[w] += 1

    #alter the dict with idf scores:
    for key, value in idf_scores.items():
        idf_scores[key] = math.log(num_docs/value, 10)

    return idf_scores


def get_corpus_topics(text, lda_model):
    '''
    :param text:
    :param lda_model:
    :return: list of document with topic IDs
    '''
    doc_topic_dist = []
    _texts = [' '.join(t for t in text)]
    texts = [simple_preprocess(doc) for doc in _texts]
    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(line) for line in texts]
    doc_topics = lda_model.get_document_topics(corpus, minimum_probability=0.0)
    for _d in doc_topics:
        doc_topic_dist.append(_d)

    return doc_topic_dist



def lda_analysis(input_data, selected_json_path, overwrite=False, num_topics = 3, num_sentences = 10):
    if os.path.exists(selected_json_path) and not overwrite:
        with open(selected_json_path) as infile:
            return json.load(infile)

    document_collection = get_document_collection(input_data)
    idf_scores = get_idf_scores(document_collection)
    picked_sentences = {}
    # treat each set of documents as a separate corpus and find topics?
    for key, value in input_data.items():
        _texts = []
        for k, v in input_data[key].items():
            _texts.append(' '.join(input_data[key][k]['lemmas']))

        texts = [simple_preprocess(doc) for doc in _texts]
        dictionary = corpora.Dictionary(texts)
        corpus = [dictionary.doc2bow(line) for line in texts]

        # build lda model:
        lda_model = LdaMulticore(corpus=corpus,
                                 id2word=dictionary,
                                 random_state=100,
                                 num_topics=num_topics,
                                 passes=10,
                                 chunksize=1000,
                                 batch=False,
                                 alpha='asymmetric',
                                 decay=0.5,
                                 offset=64,
                                 eta=None,
                                 eval_every=0,
                                 iterations=100,
                                 gamma_threshold=0.001,
                                 per_word_topics=True)

        # get document topic distribution:
        doc_topic_dist = get_corpus_topics(_texts, lda_model)

        # print(lda_model.show_topics(num_words=20))
        topic_terms = lda_model.show_topics(num_words=50)
        # get top words for each topic:
        topic_term_dict = {}
        rel_terms = []
        for topic_dist in topic_terms:
            topic_id = topic_dist[0]
            topic_term_dict[topic_id] = {}
            topic_terms = topic_dist[1]
            for _split in topic_terms.split('+'):
                topic_term_prob = _split.split('*')[0]
                topic_term = str(_split.split('*')[1]).replace('"', '').strip()
                topic_term_dict[topic_id][topic_term] = float(topic_term_prob)
                # rel_terms.append(topic_term)

        picked_sentences[key] = {}
        # pick sentences from the corpus that have highest score for the topic terms according to some score
        # this may be implemented later with the tf-idf scores
        summary_sentences = {}
        sen_ranker = []
        # calculate rank for each sentence with respect to each topic:
        for k, v in input_data[key].items():
            sen = k
            # sen = sen.lower()
            sen_length = len(sen.split(' '))
            sen_id = input_data[key][sen]['doc_id']
            if sen_length < 10:
                continue
            sen_topic = []
            # compute score for each topic:
            for topic in range(num_topics):
                rel_sen_terms = list(set(input_data[key][k]['lemmas']) & set(topic_term_dict[topic].keys()))
                # sen_score = len(rel_sen_terms)
                sen_score = 0
                for term in rel_sen_terms:
                    sen_score += idf_scores[term] * topic_term_dict[topic][term]
                sen_score = sen_score / sen_length
                sen_topic.append((topic, sen_score, sen, sen_id))

            # select top one from sen_topic and append to sen_ranker:
            top_sen_topic = sorted(sen_topic, key=lambda x: x[1], reverse=True)[0]
            sen_ranker.append(top_sen_topic)
            # sen_ranker[idx]['sen_score'] = sen_score
            # sen_ranker[idx]['sentence'] = sen

        # print(sen_ranker)
        # select max top 3 sentences only if they have a score more than 1
        sorted_sentences = sorted(sen_ranker, key=lambda x: x[1], reverse=True)[0:num_sentences]

        for _sen in sorted_sentences:
            topic = _sen[0]
            sen_score = _sen[1]
            sen = _sen[2]
            sen_id = _sen[3]
            input_data[key][sen].update({"LDAscore": sen_score})
            input_data[key][sen].update({"lda_topic_id": topic})
            summary_sentences[sen] = input_data[key][sen]

        picked_sentences[key] = summary_sentences

    with open(selected_json_path, "w") as outfile:
        json.dump(picked_sentences, outfile, indent=2)
    return picked_sentences


# MAIN #

# input data - the output from PREPROCESSING
# picked_sentences = LDA_analysis(input_data)
if __name__ == "__main__":
    # Test LDA module
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.json")
    parser.add_argument("--deliverable", type=str, default="D2", help='deliverable number, i.e. D2')
    parser.add_argument("--split", type=str, default="training", choices=["devtest", "evaltest", "training"])
    parser.add_argument("--run_id", default=None)
    parser.add_argument("--test", default=False)
    args = parser.parse_args()
    run(args)
