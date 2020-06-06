import os
import pandas as pd 
from collections import defaultdict


results = {
    "devtest": defaultdict(dict), 
    "evaltest": defaultdict(dict)
}

for f in os.listdir("../results"):
    rows = []
    if f.startswith("D4_devtest_rouge_scores_"):
        split = "devtest"
    elif f.startswith("D4_evaltest_rouge_scores_"):
        split = "evaltest"
    else:
        continue 

    num_sentences = f.split("_")[-2]
    sim_threshold = f.split("_")[-1].strip(".out")
    
    result_line = ""
    with open("../results/"+f) as infile:
        for l in infile:
            if "ROUGE-2 Average_R" in l:
                result_line = l
                break
    
    recall = result_line.split()[3]
    
    results[split][num_sentences][sim_threshold] = recall
    


num_sentences = [10, 20, 30, 40, 50]
sim_thresholds = [0.91, 0.93, 0.95, 0.97, 0.99]

rows = {
    "devtest": [], "evaltest": []
}
for split in ("devtest",): #, "evaltest"):
    for ns in num_sentences:
        row = [results[split][str(ns)][str(st)] for st in sim_thresholds]
        rows[split].append(row)

devtest_df = pd.DataFrame(rows["devtest"], columns=sim_thresholds, index=num_sentences)
evaltest_df = pd.DataFrame(rows["evaltest"], columns=sim_thresholds, index=num_sentences)