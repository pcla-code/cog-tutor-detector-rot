import pandas as pd
import argparse
import os


def extract_features(input_df: pd.DataFrame, id_column: str, output_pathname: str):
    processed_df = input_df[[id_column, 'probability_predict', 'label_predict']]
    described_df = processed_df.groupby(id_column).describe()
    described_df.columns = described_df.columns.to_series().apply(lambda x: "_".join(x))
    described_df.to_csv(output_pathname)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='Extract features from gaming predictions in MATHia')
    ap.add_argument('mathia_csv', help='Path to gaming predictions (CSV or TSV) to use as input')
    ap.add_argument('user_id', help='column name for student/user ID')
    ap.add_argument('output_dir', help='Output directory path')
    args = ap.parse_args()

    print('Loading')
    sep = '\t' if args.features_csv.endswith('.tsv') else ','
    predictions_df = pd.read_csv(args.mathia_csv)
    student_id = args.user_id
    output_csv = args.output_dir
    filename = os.path.basename(args.features_csv)

    extract_features(predictions_df, student_id, output_csv)

