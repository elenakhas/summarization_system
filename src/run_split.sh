#!/bin/sh
source /home2/schan2/anaconda3/etc/profile.d/conda.sh
conda activate /home2/schan2/anaconda3/envs/573
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home2/schan2/anaconda3/lib/

# cd src/
SPLIT=$1
SIM_THRESHOLD=$2

python3 run_pipeline.py --deliverable "D4" --split $SPLIT --run_id "D4test"  \
--model_name "bert-base-cased-finetuned-mrpc" \
--use_embeddings --sim_threshold $SIM_THRESHOLD --num_sentences 18
perl /dropbox/19-20/573/code/ROUGE/ROUGE-1.5.5.pl -e /dropbox/19-20/573/code/ROUGE/data -a -n 2 -x -m -c 95 -r 1000 -f A -p 0.5 -t 0 -l 100 -s -d $PWD/rouge_run_D4_${SPLIT}.xml > $PWD/../results/D4_${SPLIT}_rouge_scores.out
