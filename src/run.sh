#!/bin/sh
source /home2/schan2/anaconda3/etc/profile.d/conda.sh
conda activate /home2/schan2/anaconda3/envs/573
cd src/
python3 -m spacy download en_core_web_lg 
python3 run_pipeline.py --split "devtest" --run_id "D3run0"
perl /dropbox/19-20/573/code/ROUGE/ROUGE-1.5.5.pl -e /dropbox/19-20/573/code/ROUGE/data -a -n 2 -x -m -c 95 -r 1000 -f A -p 0.5 -t 0 -l 100 -s -d $PWD/rouge_run_D2_devtest.xml > $PWD/../results/D2_rouge_scores.out
