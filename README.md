# CogTutorGamingDetectors
Detecting students exhibiting "Gaming The System" behavior and predicting year end test scores based on these behaviors.


## Step 1: Extracting features from raw logfiles
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
```
python extract_features.py ./path/to/logfile.csv ./extractedFeaturesDir  --concat ./extractedFeaturesDir/final_features_file.csv
```


## Step 2: Running the gaming detector

The second step is to run the gaming detector using the features extracted in step one. This
is done using 'modeling_gaming.py`. This file takes command line args with both required and optional
flags.

* `features_csv` (required): Path to extracted features file (CSV or TSV) to use as input
* `train_label` (required): column name of training labels
* `user_id` (required): column name for student/user ID
* `output_dir` (required): Path to where to write CSV output file, including name (but not including '.csv'). Each model run will save as a unique file.
* `--training_data_csv` (optional): Path to training data. If provided, all of the features csv will be used for testing

For example, you might run the command
``` 
python modeling_gaming.py ./path/to/final_features_file.csv "label" "user_id" ./extractedFeaturesDir/YEAR_SCHOOL_gaming_predictions --training_data_csv training_data.csv
```

## Step 3: Extract features from gaming predictions

The third step is to extract features from the predictions created by the gaming detector. This is done using `extract_features_from_gaming_predictions.py`. This file takes command line args with required flags. Only one output file will be created.

* `features_csv` (required): Path to gaming predictions (CSV or TSV) to use as input
* `user_id` (required): column name for student/user ID
* `output_dir` (required): Path to where to write CSV output file, including name.

For example, you might run the command
```
python extract_features_from_gaming_predictions.py ./path/to/gaming_predictions.csv "user_id" ./path/to/final_features_file.csv
```


## Step 4: Running the test score predictive models

The fourth step is to predict test scores using the features extracted in step three. This
is done using 'predict_test_scores.py`. This file takes command line args with required flags.

* `features_csv` (required): Path to extracted features file (CSV or TSV) to use as input
* `train_label` (required): column name of training labels
* `user_id` (required): column name for student/user ID
* `output_dir` (required): Path to where to write CSV output file, including name (but not including '.csv'). Each model run will save as a unique file.

For example, you might run the command
```
python modeling_gaming.py ./path/to/final_features_file.csv "label" "user_id" ./extractedFeaturesDir/YEAR_SCHOOL_test_score_predictions
```
