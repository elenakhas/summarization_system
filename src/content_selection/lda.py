import os
import sys
import json
import gensim
from gensim import corpora
from gensim.utils import simple_preprocess
from gensim.models import LdaModel, LdaMulticore


def lda_analysis(input_data, selected_json_path, overwrite=False):
    if os.path.exists(selected_json_path) and not overwrite:
        with open(selected_json_path) as infile:
            return json.load(infile)
    
    picked_sentences = {}
    #treat each set of documents as a separate corpus and find topics?
    for key, value in input_data.items():
        _texts = []
        for k, v in input_data[key].items():
            _texts.append(' '.join(input_data[key][k]['lemmas']))

        texts = [simple_preprocess(doc) for doc in _texts]
        dictionary = corpora.Dictionary(texts)
        corpus = [dictionary.doc2bow(line) for line in texts]

        #build lda model:
        lda_model = LdaMulticore(corpus=corpus,
                                 id2word=dictionary,
                                 random_state=100,
                                 num_topics=3,
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


        topic_terms = lda_model.show_topics()
        #get top words for each topic:
        topic_term_dict = {}
        rel_terms = []
        for topic_dist in topic_terms:
            topic_id = topic_dist[0]
            topic_term_dict[topic_id] = {}
            topic_terms = topic_dist[1]
            for _split in topic_terms.split('+'):
                topic_term_prob = _split.split('*')[0]
                topic_term = str(_split.split('*')[1]).replace('"','').strip()
                topic_term_dict[topic_id][topic_term] = topic_term_prob
                rel_terms.append(topic_term)

        picked_sentences[key] = {}
        #pick sentences from the corpus that have highest score for the topic terms according to some score
        #this may be implemented later with the tf-idf scores
        summary_sentences = {}
        sen_ranker = []
        for k,v in input_data[key].items():
            sen = k.lower()
            rel_sen_terms = list(set(input_data[key][k]['lemmas']) &  set(rel_terms))
            sen_score = len(rel_sen_terms)

            # IF YOU WANT ALL SENTENCES AND THEIR RANKS, UNCOMMENT THIS LINE, COMMENT EVERYTHING UNTIL THE END AND RETURN input_data
#             v.update({"LDAscore": sen_score})


#             # IF YOU WANT ALL ONLY TOP N - USE THIS TILL THE END AND RETURN picked_sentences           
            sen_ranker.append((sen_score, k))

        sorted_sentences = sorted(sen_ranker, key = lambda x : x[0], reverse = True)[0:10]

        for _sen in sorted_sentences:
            sent = _sen[1]
            score = _sen[0]
            input_data[key][sent].update({"LDAscore": score})
            summary_sentences[sent] = input_data[key][sent]

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
    
