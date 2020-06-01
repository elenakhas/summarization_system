#!/bin/sh
cd src/
bash run_split.sh "training" 0.94
bash run_split.sh "devtest" 0.94
bash run_split.sh "evaltest" 0.94
