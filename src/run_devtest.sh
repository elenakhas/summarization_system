#!/bin/sh
source /home2/schan2/anaconda3/etc/profile.d/conda.sh
conda activate /home2/schan2/anaconda3/envs/573
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home2/schan2/anaconda3/lib/

cd src/
SPLIT="devtest"
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home2/schan2/anaconda3/lib/
python3 run_pipeline.py --deliverable "D4" --split $SPLIT \
--model_name "bert-base-cased-finetuned-mrpc" \
--use_embeddings 