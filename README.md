# summarization_system
A group project for NLP Systems and Applications @ UW

# D2

To run the summarization system on [patas](https://wiki.ling.washington.edu/bin/view.cgi):
```
condor_submit D2.cmd
```

To run end to end with training files:
```
source /home2/schan2/anaconda3/etc/profile.d/conda.sh
conda activate /home2/schan2/anaconda3/envs/573
python3 run_pipeline.py --split training
```

To run end to end with test files:
```
source /home2/schan2/anaconda3/etc/profile.d/conda.sh
conda activate /home2/schan2/anaconda3/envs/573
python3 run_pipeline.py --split devtest 
```



# Contributors
* Erica Gardner 
* Saumya Shah 
* Vikash Kumar 
* Elena Khasanova
* Sophia Chan 

