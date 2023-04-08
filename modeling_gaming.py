import xgboost as xgb
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import GroupKFold
from sklearn.dummy import DummyClassifier
from sklearn.base import BaseEstimator
from imblearn.over_sampling import SMOTE
import argparse
import os
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


def preprocess_data(training_data: pd.DataFrame, label: str) -> pd.DataFrame:
    training_data = training_data[training_data[label] != '?']
    training_data[label] = (training_data[label] == 'G').astype(int)
    training_data = training_data.fillna(0)
    return training_data


def train_model(feat_df: pd.DataFrame, group_id: str, label: str, classifier: BaseEstimator, num_splits: int = 4, sm: bool = True):
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

        train_X = train_inst[features].copy()
        train_y = train_inst[label].copy()

        test_x = test_inst[features].copy()
        test_y = test_inst[label].copy()

        if sm:
            sm = SMOTE(random_state=42)
            train_X, train_y = sm.fit_resample(train_X, train_y)

        classifier.fit(train_X, train_y)
        probs = classifier.predict_proba(test_x)
        label_predict = classifier.predict(test_x)
        cur_preds['probability_predict'].iloc[test_indices] = probs[:, 1]
        cur_preds['label_predict'].iloc[test_indices] = label_predict

    return cur_preds


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='Build predictive models for gaming the system in MATHia')
    ap.add_argument('features_csv', help='Path to extracted features file (CSV or TSV) to use as input')
    ap.add_argument('train_label', help='column name of ground truth labels')
    ap.add_argument('user_id', help='column name for student/user ID')
    args = ap.parse_args()

    print('Loading')
    sep = '\t' if args.features_csv.endswith('.tsv') else ','
    df = pd.read_csv(args.features_csv)
    training_label = args.train_label
    student_id = args.user_id
    filename = os.path.basename(args.features_csv)

    processed_df = preprocess_data(df, training_label)

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
        xgb.XGBClassifier(random_state=i, use_label_encoder=False, eval_metric='logloss')
    ]

    for name, clf in zip(names, classifiers):
        # new to new predictions
        final_preds = train_model(processed_df, student_id, training_label, clf)
        final_preds['orig_file'] = filename
        final_preds.to_csv(str(name) + '_gaming_predictions_pasadena_2021_2022.csv', index=False)
