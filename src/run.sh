#!/bin/sh
source /home2/schan2/anaconda3/etc/profile.d/conda.sh
conda activate /home2/schan2/anaconda3/envs/573
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home2/schan2/anaconda3/lib/
cd src/
python3 -m spacy download en_core_web_lg

# Run the pipeline on training (for testing)
# python3 run_pipeline.py --split "training" --run_id "D3test"  \
# --model_name "bert-base-cased-finetuned-mrpc" \
# --use_embeddings --sim_threshold 0.97 --num_sentences 20

# Run the pipeline on devtest data
python3 run_pipeline.py --split "devtest" --run_id "D3test"  \
--model_name "bert-base-cased-finetuned-mrpc" \
--use_embeddings --sim_threshold 0.97 --num_sentences 20

# Run the pipeline on evaltest data
python3 run_pipeline.py --split "evaltest" --run_id "D3eval"  \
--model_name "bert-base-cased-finetuned-mrpc" \
--use_embeddings --sim_threshold 0.97 --num_sentences 20

# Evaluate devtest summaries
perl /dropbox/19-20/573/code/ROUGE/ROUGE-1.5.5.pl -e /dropbox/19-20/573/code/ROUGE/data \
-a -n 2 -x -m -c 95 -r 1000 -f A -p 0.5 -t 0 -l 100 -s -d \
$PWD/rouge_run_D4_devtest.xml > $PWD/../results/D4_devtest_rouge_scores.out

# Evaluate evaltest summaries
perl /dropbox/19-20/573/code/ROUGE/ROUGE-1.5.5.pl -e /dropbox/19-20/573/code/ROUGE/data \
-a -n 2 -x -m -c 95 -r 1000 -f A -p 0.5 -t 0 -l 100 -s -d \
$PWD/rouge_run_D4_evaltest.xml > $PWD/../results/D4_evaltest_rouge_scores.out
