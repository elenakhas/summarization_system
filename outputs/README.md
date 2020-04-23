# Outputs

Contains summaries based on running base summarization system on `devtest` data files.
The output names have the following format:
* Given topic ID e.g. `D0901A`
* Split into:
    * id_part1 = `D0901`, and
    * id_part2 = `A`
* Output file name should be:
    * `[id_part1]-A.M.100.[id_part2].[some_unique_alphanum]`

The output names must match the peer names in ROUGE config.
