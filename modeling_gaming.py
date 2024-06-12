import xgboost as xgb
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
import argparse
import os
import warnings


def preprocess_data(training_data: pd.DataFrame, label: str) -> pd.DataFrame:
    training_data = training_data[training_data[label] != '?'].copy()
    training_data[label] = (training_data[label] == 'G').astype(int)
    training_data = training_data.fillna(0)
    return training_data


def train_predict_model(train_df: pd.DataFrame, feat_df: pd.DataFrame, group_id: str, label: str, pipe: Pipeline, sm: bool = True):
    features = list(train_df.columns)
    features.remove(label)
    features.remove(group_id)
    features.remove('orig_index')
    predictions = feat_df.copy(deep=True)
    predictions = predictions.reindex(columns=['orig_index', group_id, label, 'probability_predict', 'label_predict'])


    if sm:
        sm = SMOTE(random_state=42)
        train_x, train_y = sm.fit_resample(train_df[features], train_df[label])
    else:
        train_x, train_y = train_df[features], train_df[label]

    pipe.fit(train_x, train_y)
    probs_predict = pipe.predict_proba(feat_df[features])
    label_predict = pipe.predict(feat_df[features])
    predictions['probability_predict'] = probs_predict[:, 1]
    predictions['label_predict'] = label_predict

    return predictions


def train_predict_model_with_splits(feat_df: pd.DataFrame, group_id: str, label: str, pipe: Pipeline, num_splits: int = 4, sm: bool = True):
    features = list(feat_df.columns)
    features.remove(label)
    features.remove(group_id)
    features.remove('orig_index')
    groups = feat_df[group_id]
    cur_preds = feat_df.copy(deep=True)
    cur_preds = cur_preds.reindex(columns=['orig_index', group_id, label, 'probability_predict', 'label_predict'])

    for iter_num, fold_indices in enumerate(GroupKFold(n_splits=num_splits).split(X=feat_df, groups=groups)):
        train_indices = fold_indices[0]
        test_indices = fold_indices[1]

        train_inst = feat_df.iloc[train_indices].copy()
        test_inst = feat_df.iloc[test_indices].copy()

        train_x = train_inst[features].copy()
        train_y = train_inst[label].copy()

        test_x = test_inst[features].copy()
        test_y = test_inst[label].copy()

        if sm:
            sm = SMOTE(random_state=42)
            train_x, train_y = sm.fit_resample(train_x, train_y)

        pipe.fit(train_x, train_y)
        probs = pipe.predict_proba(test_x)
        label_predict = pipe.predict(test_x)
        cur_preds['probability_predict'].iloc[test_indices] = probs[:, 1]
        cur_preds['label_predict'].iloc[test_indices] = label_predict

    return cur_preds


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='Build predictive models for gaming the system in MATHia')
    ap.add_argument('features_csv', help='Path to extracted features file (CSV or TSV) to use as input')
    ap.add_argument('train_label', help='column name of training labels')
    ap.add_argument('user_id', help='column name for student/user ID')
    ap.add_argument('output_dir', help='Path to where to write CSV output file')
    ap.add_argument('--training_data_csv',
                    help='Path to training data. If provided, all of the features csv will be used for testing.')
    args = ap.parse_args()

    print('Loading')
    sep = '\t' if args.features_csv.endswith('.tsv') else ','
    features_df = pd.read_csv(args.features_csv)
    training_label = args.train_label
    student_id = args.user_id
    output_csv = args.output_dir
    filename = os.path.basename(args.features_csv)

    if args.training_data_csv:
        print('Training data provided. All data in the features_csv will be used for testing.')
        training_df = pd.read_csv(args.training_data_csv)
        training_df = preprocess_data(training_df, training_label)
        features_df = features_df.fillna(0)


    else:
        features_df = preprocess_data(features_df, training_label)



    i = 0
    names = [
        "Dummy",
        "Decision Tree",
        "Random Forest",
        "Neural Net",
        "XGBoost"
    ]
    classifiers = [
        DummyClassifier(strategy="stratified"),
        DecisionTreeClassifier(max_depth=5, random_state=i),
        RandomForestClassifier(max_depth=5, random_state=i),
        MLPClassifier(random_state=i),
        xgb.XGBClassifier(random_state=i, eval_metric='logloss')
    ]

    for name, clf in zip(names, classifiers):

        pipeline = Pipeline([('name', clf)])

        # if training data was provided, train model and then test on extracted features
        if args.training_data_csv:
             final_preds = train_predict_model(training_df, features_df, student_id, training_label, pipeline)

        # new to new predictions (if no training data provided)
        else:
            final_preds = train_predict_model_with_splits(features_df, student_id, training_label, pipeline)

        final_preds['orig_file'] = filename
        final_preds.to_csv(str(output_csv + '_' + name) + '.csv', index=False)
