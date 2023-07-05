import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
import argparse
import os
from sklearn.model_selection import GridSearchCV


#can replace this whole function with crosval predict
def train_predict_model_with_splits(feat_df: pd.DataFrame, group_id: str, label: str, pipe: Pipeline, num_splits: int = 4, sm: bool = True):
    features = list(feat_df.columns)
    features.remove(label)
    features.remove(group_id)
    cur_preds = feat_df.copy(deep=True)
    cur_preds = cur_preds.reindex(columns=[group_id, label, 'predicted_MCAS'])

    for iter_num, fold_indices in enumerate(KFold(n_splits=num_splits).split(X=feat_df)):
        train_indices = fold_indices[0]
        test_indices = fold_indices[1]

        train_inst = feat_df.iloc[train_indices].copy()
        test_inst = feat_df.iloc[test_indices].copy()

        train_x = train_inst[features].copy()
        train_y = train_inst[label].copy()

        test_x = test_inst[features].copy()
        test_y = test_inst[label].copy()

#         if sm:
#             sm = SMOTE(random_state=42)
#             train_x, train_y = sm.fit_resample(train_x, train_y)

        pipe.fit(train_x, train_y)
        label_predict = pipe.predict(test_x)
        cur_preds['predicted_MCAS'].iloc[test_indices] = label_predict

    return cur_preds


def preprocess_data(training_data: pd.DataFrame, label: str) -> pd.DataFrame:
    training_data = training_data[~training_data[label].isna()].copy()
    training_data = training_data.fillna(0)
    return training_data


if __name__ == '__main__':
    csv_name = "/Users/clarabelitz/Library/CloudStorage/Box-Box/NSF ECR Mathia algorithmic bias/data/analysis/MCAS predictions from gaming/Feature Extraction from Gaming/features_for_mcas_prediction.csv"
    predictive_features = pd.read_csv(csv_name)

    label_name = "MCAS"
    student_id = "user_id"

    features = list(predictive_features.columns)
    features.remove(student_id)
    features.remove(label_name)

    processed_training_df = preprocess_data(predictive_features, label_name)

    y = processed_training_df[label_name]
    X = processed_training_df[features]

    i = 0
    names = [
    "Decision Tree",
    "Random Forest",
    "Neural Net",
    "XGBoost"
    ]
    # use regression for MCAS since it is continuous
    regressors = [
        DecisionTreeRegressor(max_depth=5, random_state=i),
        RandomForestRegressor(max_depth=5, random_state=i),
        MLPRegressor(random_state=i),
        xgb.XGBRegressor(random_state=i, eval_metric='logloss')
    ]

    # for name, clf in zip(names, regressors):
    #     pipeline = Pipeline([('name', clf)])
    #
    #     final_preds = train_predict_model_with_splits(processed_training_df, student_id, label_name, pipeline, sm=False)
    #
    #     final_preds.to_csv(str(name) + '_MCAS_predictions_all_brockton_21_22_regressor_test.csv', index=False)
    #     r2score = r2_score(final_preds['MCAS'], final_preds['predicted_MCAS'])
    #     print(name, r2score)

    # to-do: make this dictionary of names and models (and parameters) rather than two lists
    gs_names = [
        "Random Forest"
    ]

    param_grid = {
        "model__min_samples_leaf": [1, 2, 4, 8, 16, 32],
        "model__max_features": ["sqrt", "log2", .1, .25, .5, .75, 0.9, 1.0]
    }

    for name in gs_names:
        all_predictions = []
        for i in range(10):
            # can do a name check for each regressor
            if name == "Random Forest":
                reg = RandomForestRegressor(random_state=i)
                print(name)

            pipeline = Pipeline([('model', reg)])
            search = GridSearchCV(pipeline, param_grid, cv=10, n_jobs=2)

            predictions = cross_val_predict(search, X, y, cv=10)
            # print("Best parameter (CV score=%0.3f):" % search.best_score_)
            # print(search.best_params_)
            all_predictions.append(predictions)

        average_pred = np.mean(all_predictions)  # check that this is averaged in correct direction
        print(average_pred)
        pd.Series(average_pred).to_csv(str(name) + '_MCAS_predictions_all_brockton_21_22_gs_test.csv', index=False)
