import pandas as pd
import numpy as np
import os
import argparse


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='Build long-form dataset of predictions for gaming the system in MATHia')
    ap.add_argument('directory', help='Path to directory with prediction files (CSV or TSV) to use as input')
    ap.add_argument('label', help='column name of ground truth labels')
    ap.add_argument('predict', help='column name of predicted  labels')
    ap.add_argument('user_id', help='column name for student/user ID')
    ap.add_argument('output_dir', help='Output directory path')
    args = ap.parse_args()

    directory = args.directory
    user_id = args.user_id
    label = args.label
    predict = args.predict
    output_dir = args.output_dir

    models_dict = {'dummy': 'dm',
                   'decision tree': 'dt',
                   'random forest': 'rf',
                   'neural net': 'nn',
                   'xgboost': 'xgb'
                   }

    col_list = ['model', user_id, 'correct', 'orig_file']

    all_predictions = pd.DataFrame(columns=col_list)

    # iterate over files in the directory
    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)
        # checking if it is a file
        if os.path.isfile(f):
            cur_file = pd.read_csv(f)
            temp_df = cur_file.copy(deep=True)
            temp_df = temp_df.reindex(columns=col_list)
            temp_df['correct'] = np.where(cur_file[label] == cur_file[predict], 1, 0)
            for key in models_dict.keys():
                if key in str.lower(filename):
                    model_type = models_dict[key]
                    temp_df['model'] = model_type

            all_predictions = pd.concat([all_predictions, temp_df], ignore_index=True)

    all_predictions.to_csv(os.path.join(output_dir, 'all_models_predictions_correctness.csv'), index=False)

