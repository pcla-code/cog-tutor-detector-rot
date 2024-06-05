# CogTutorGamingDetectors
Detecting students exhibiting "Gaming The System" behavior.


## Extracting features from raw logfiles
The first step for the gaming detector requires feature extraction from the raw MATHia (or other datashop formatted) logifles.
This is done using `extract_features.py`. This file takes command line args with both required and optional flags.

* `mathia_csv` (required): Path to MATHia log file (CSV or TSV) to use as input
* `output_dir` (required): Output directory path
* `--labels` (optional): Path to space-separated labels TXT file for supervised extraction 
* `--concat` (optional): Path to final concatenated output CSV file

By default, each student's extracted features will be saved to a unique CSV file in the output directory, `STU_ID.csv`.
The `--concat` flag is used to indicate that you also wish to save a version of the file with all of the concatenated data.
This should include a full path, including file name, to which the data will be saved.

For example, you might run the command
```python extract_features.py ./path/to/logfile.csv ./extractedFeaturesDir  --concat ./extractedFeaturesDir/final_file.csv```
