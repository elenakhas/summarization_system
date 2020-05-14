import torch
from tqdm import tqdm 
from transformers import BertTokenizer, BertModel
# from transformers import AutoTokenizer, AutoModelWithLMHead
import numpy as np
from scipy.spatial.distance import cosine
import pandas as pd
import os
import random 


def tokenize(sentence_list, model_name="bert-base-cased"):
    tokenizer = BertTokenizer.from_pretrained(model_name)
    inputs_list = []
    for sentence in tqdm(sentence_list, desc="tokenizing"):
        input_ids = torch.tensor(tokenizer.encode(sentence)).unsqueeze(0)
        inputs_list.append(input_ids)
    return inputs_list


def make_embeddings(sentence_list, model_name="bert-base-cased"):
    """
    Given a list of sentences, return their embeddings.
    The embeddings are a mean over the last hidden layer of 
    a pretrained BERT model.
    """
    inputs_list = tokenize(sentence_list)
    model = BertModel.from_pretrained(model_name)
    sentence_embeddings = []
    for sentence, input_ids in tqdm(zip(sentence_list, inputs_list), desc="embedding"):
        with torch.no_grad():
            outputs = model(input_ids)
            embed = torch.mean(outputs[0], dim=0)
            embed = torch.mean(embed, dim=0)
            sentence_embeddings.append(embed)
    return sentence_embeddings


def test():
    lst = [
        "This is a sentence.",
        "This is another sentence that is much much longer and should be different."
    ]
    
    embeddings = make_embeddings(lst)
    print("cosine between lst[0] and lst[1]")
    print(embeddings[0].shape)
    print(1 - cosine(embeddings[0], embeddings[1]))

    print("cosine between lst[0] and lst[0]")
    print(1 - cosine(embeddings[0], embeddings[0]))

    print("cosine between lst[1] and lst[1]")
    print(1 - cosine(embeddings[1], embeddings[1]))


if __name__ == "__main__":
    models = (
        "bert-base-cased",
        "bert-base-cased-finetuned-mrpc",
    )
    
    df = pd.read_csv(os.path.join("working_files", "cosine_experiments", 
        "candidates.csv"), header=None)
    s1_list = list(df[1])
    s2_list = list(df[2])

    indices = random.sample(range(len(s1_list)), 100)
    s1_list = [s for i, s in enumerate(s1_list) if i in indices]
    s2_list = [s for i, s in enumerate(s2_list) if i in indices]
    s1_embeds = make_embeddings(s1_list)
    s2_embeds = make_embeddings(s2_list)

    rows = []
    for i, (s1, s2) in enumerate(zip(s1_list, s2_list)):
        embed1 = s1_embeds[i]
        embed2 = s2_embeds[i]
        # print(1 - cosine(embed1, embed2))
        rows.append([s1, s2, 1 - cosine(embed1, embed2)])
    
    df = pd.DataFrame(rows)
    df = df.sort_values(by=2, ascending=False)
    df.to_csv(os.path.join("working_files", "cosine_experiments", 
        "candidates_base.csv"))

