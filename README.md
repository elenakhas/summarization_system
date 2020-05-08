# summarization_system
A group project for NLP Systems and Applications @ UW

# D3

To run the summarization system on **condor**:
```
condor_submit D3.cmd
```



To run on **patas**:
1. Activate conda environment
    ```
    source /home2/schan2/anaconda3/etc/profile.d/conda.sh
    conda activate /home2/schan2/anaconda3/envs/573
    ```
2. Run the pipeline, replacing `<split>` with either `training` or `devtest`
    ```
    python3 run_pipeline.py --split <split>
    ```
3. To test your changes, you can run the pipeline on a subset of 3 topics in the `devtest` data
    ```
    python3 run_pipeline.py --split devtest --test
    ```


# Cached outputs
The outputs of the following modules are cached under `src/working_files`:
* Data loading
* Preprocessing
* LDA

If your changes to a module change the output, you need to set `overwrite=True` for that module in `src/run_pipeline.py`. For example, to recreate output from the preprocessing step:
```python
preprocessed_data = preprocess(input_data, os.path.join(
        data_store["working_dir"], os.path.basename(xml_filename)[:-4] + ".json.preprocessed"),
        overwrite=True)
```


# Contributors
* Erica Gardner 
* Saumya Shah 
* Vikash Kumar 
* Elena Khasanova
* Sophia Chan 

