import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn import linear_model, naive_bayes, preprocessing
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
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
    cur_preds = cur_preds.reindex(columns=[group_id, label, 'predicted_score'])

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
        cur_preds['predicted_score'].iloc[test_indices] = label_predict

    return cur_preds


def preprocess_data(training_data: pd.DataFrame, label: str) -> pd.DataFrame:
    training_data = training_data[~training_data[label].isna()].copy()
    training_data = training_data.fillna(0)
    return training_data


if __name__ == '__main__':
    csv_name = "path/to/features_file.csv"
    predictive_features = pd.read_csv(csv_name)

    label_name = "test_score"
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
    # use regression for test score prediction since it is continuous
    regressors = [
        DecisionTreeRegressor(max_depth=5, random_state=i),
        RandomForestRegressor(max_depth=5, random_state=i),
        MLPRegressor(random_state=i),
        xgb.XGBRegressor(random_state=i, eval_metric='logloss')
    ]

    gs_names = [
        "Random Forest",
        "Extra Trees",
        "Decision Tree",
        "XGBoost",
        "GNB",
        "Linear Regression"
    ]

    # parameter grid for tree based models
    trees_param_grid = {
        "model__min_samples_leaf": [1, 2, 4, 8, 16, 32],
        "model__max_features": ["sqrt", "log2", .1, .25, .5, .75, 0.9, 1.0]
    }

    # parameter grid for XGB, matches tree based models
    xgb_param_grid = {
        "model__min_child_weight": [1, 2, 4, 8, 16, 32],
        "model__colsample_bytree": [.1, .25, .5, .75, 0.9, 1.0]
    }

    # empty object for models without hyperparameters
    empty_param_grid = None

    for name in gs_names:
        print("Training " + name)
        all_predictions = []
        for i in range(10):
            # can do a name check for each regressor
            if name == "Decision Tree":
                reg = DecisionTreeRegressor(random_state=i)
                param_grid = trees_param_grid

            elif name == "Random Forest":
                reg = RandomForestRegressor(random_state=i)
                param_grid = trees_param_grid

            elif name == "Extra Trees":
                reg = ExtraTreesRegressor(random_state=i)
                param_grid = trees_param_grid

            elif name == "XGBoost":
                reg = xgb.XGBRegressor(random_state=i)
                param_grid = xgb_param_grid

            elif name == "GNB":
                reg = naive_bayes.GaussianNB()
                param_grid = empty_param_grid
                scaler = preprocessing.StandardScaler()
                X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
                y = pd.Series(y)

            elif name == "Linear Regression":
                reg = linear_model.LinearRegression()
                param_grid = empty_param_grid

            if param_grid is None:
                # no parameters to search, just get predictions
                predictions = cross_val_predict(reg, X, y, cv=10)

            else:
                # use gridsearch to optimize hyperparameters
                pipeline = Pipeline([('model', reg)])
                search = GridSearchCV(pipeline, param_grid, cv=10, n_jobs=2)

                predictions = cross_val_predict(search, X, y, cv=10)

            # print("Best parameter (CV score=%0.3f):" % search.best_score_)
            # print(search.best_params_)
            all_predictions.append(predictions)

        average_pred = np.mean(all_predictions, axis=0)  # average predictions per student
        average_df = pd.DataFrame(index=processed_training_df[student_id], data=average_pred, columns=['average_score_prediction'])
        average_df.to_csv(str(name) + '_score_predictions_averaged_all_students_21_22.csv')
