import os
import json
import argparse
from gensim import corpora
from gensim.utils import simple_preprocess
from gensim.models import LdaModel
from collections import OrderedDict


# reaad json file
def parseJson(json_file):
    '''
    parsing the JSON file from the pre-processing pipeline
    :param json_file
    :return: dictionary with docID as key
    '''

    with open(json_file) as f:
        data = json.load(f)

    return data

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


def lda_analysis(input_data, num_topics=3, random_state=1):
    
    # treat each set of documents as a separate corpus and find topics?
    for key, value in input_data.items():
        _texts = []
        for k, v in input_data[key].items():
            _texts.append(' '.join(input_data[key][k]['lemmas']))

        texts = [simple_preprocess(doc) for doc in _texts]
        dictionary = corpora.Dictionary(texts)
        corpus = [dictionary.doc2bow(line) for line in texts]

        # build lda model:
        lda_model = LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics, random_state=random_state)

        # get document topic distribution:
        doc_topic_dist = get_corpus_topics(_texts, lda_model)

        topic_terms = lda_model.show_topics(num_words=100)
        # get top words for each topic:
        topic_term_dict = OrderedDict()
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

        summary_sentences = {}
        sen_ranker = []
        # calculate rank for each sentence with respect to each topic:
        for k, v in input_data[key].items():
            sen = k
            # sen = sen.lower()
            sen_length = len(sen.split(' '))
            sen_id = input_data[key][sen]['doc_id']
            if sen_length <= 7:
                continue
            sen_topic = []
            # compute score for each topic:
            for topic in range(num_topics):
                rel_sen_terms = list(set(input_data[key][k]['lemmas']) & set(topic_term_dict[topic].keys()))
                sen_score = 0
                for term in rel_sen_terms:
                    sen_score += topic_term_dict[topic][term]

                sen_topic.append((topic, sen_score, sen, sen_id))

            # select top one from sen_topic and append to sen_ranker:
            top_sen_topic = sorted(sen_topic, key=lambda x: x[1], reverse=True)[0]
            sen_ranker.append(top_sen_topic)

        for _sen in sen_ranker:
            topic = _sen[0]
            sen_score = _sen[1]
            sen = _sen[2]
            sen_id = _sen[3]
            input_data[key][sen].update({"LDAscore": sen_score})
            input_data[key][sen].update({"lda_topic_id": topic})

    return input_data


def update_scores(dic):
    '''
    Updates the sentence scores in the dictionary by combining tf-idf, concreteness and LDA scoring
    '''
    new_dict = {}

    for topic_id, sent in dic.items():
        new_dict[topic_id] = dict()
        tf_idf = []
        concreteness = []
        #lda = []
        for key, info in sent.items():
            tf_idf.append(info['tf_idf'])
            concreteness.append(info['concreteness'])
            #try:
                #lda.append(info['LDAscore'])
            #except KeyError:
                #continue

        m_tf_idf = max(tf_idf)
        m_concrete = max(concreteness)
        #m_lda = 1

        for key, info in sent.items():

            if info['length'] > 7:
                info['tf_idf'] = info['tf_idf'] / m_tf_idf
                info['concreteness'] = info['concreteness'] / m_concrete
                try:
                    #info['LDAscore'] = info['LDAscore'] / m_lda
                    info['total'] = (info['tf_idf'] * info['concreteness'] * info['LDAscore']) / info['length']
                    sent_info = {k: v for k, v in info.items()}
                    new_dict[topic_id][key] = sent_info

                except KeyError:
                    continue
    return new_dict


def select_sent(data, num_sentences):
    picked_sent = {}

    for topic_id, sent in data.items():
        candidates = []
        group_1 = []
        group_2 = []
        group_3 = []
        for key, info in sent.items():
            try:
                total = info['total']
                if info['lda_topic_id'] == 0:
                    group_1.append((key, total))
                elif info['lda_topic_id'] == 1:
                    group_2.append((key, total))
                else:
                    group_3.append((key, total))
            #                     candidates.append((key, total))
            except KeyError:
                continue

        sorted_1 = sorted(group_1, key=lambda x: x[1], reverse=True)[:int(num_sentences / 3)]
        sorted_2 = sorted(group_2, key=lambda x: x[1], reverse=True)[:int(num_sentences / 3)]
        sorted_3 = sorted(group_3, key=lambda x: x[1], reverse=True)[:int(num_sentences / 3)]

        sorted_sentences = sorted_1 + sorted_2 + sorted_3

        picked_sent[topic_id] = dict()
        for sentence, score in sorted_sentences:
            sent_info = data[topic_id][sentence]
            sent_info['total'] = score
            picked_sent[topic_id][sentence] = sent_info
    return picked_sent


def sentence_selection_wrapper(input_data, selected_json_path, num_sentences=20, overwrite=False, random_state=1):
    if os.path.exists(selected_json_path) and not overwrite:
        with open(selected_json_path) as infile:
            return json.load(infile)

    new_dict = lda_analysis(input_data, random_state=random_state, num_topics=3)
    update_and_normalize = update_scores(new_dict)
    picked_sentences = select_sent(update_and_normalize, num_sentences)
    with open(selected_json_path, "w") as outfile:
        json.dump(picked_sentences, outfile, indent=2)
    return picked_sentences


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