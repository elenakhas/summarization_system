#!/bin/sh
source activate /home2/schan2/anaconda3/envs/573
#cd src/
python3 data_loader.py
perl /dropbox/19-20/573/code/ROUGE/ROUGE-1.5.5.pl -e /dropbox/19-20/573/code/ROUGE/data -a -n 2 -x -m -c 95 -r 1000 -f A -p 0.5 -t 0 -l 100 -s -d rouge_run.xml > ../results/D2_rouge_scores.out
