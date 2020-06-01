#!/bin/sh
cd src/
bash run_split.sh "training"
bash run_split.sh "devtest"
bash run_split.sh "evaltest"
