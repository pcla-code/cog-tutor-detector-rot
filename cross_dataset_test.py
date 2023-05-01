import argparse
import os

import pandas as pd
from sklearn import ensemble, metrics, model_selection


ap = argparse.ArgumentParser(description='Check if multiple datasets are better for training than '
                             'only one (e.g., across school districts)')
ap.add_argument('datasets', nargs='+',
                help='Paths to CSV datasets with matching columns. Each filename must be unique.')
args = ap.parse_args()

print('Loading/preprocessing')
dfs = {}
for fname in args.datasets:
    key = os.path.split(fname)[1]
    df = pd.read_csv(fname)
    df.drop(df[df.label == '?'].index, inplace=True)
    df.fillna(-1, inplace=True)
    df['label'] = (df.label == 'G').astype(int)
    dfs[key] = df
features = [f for f in df.columns if f not in ['orig_index', 'user_id', 'label']]


print('Running models')
for test_name in dfs:
    print()
    test_df = dfs[test_name]
    candidate_names = [k for k in dfs.keys() if k != test_name]
    for train_names in [[candidate_names[0]], [candidate_names[1]], candidate_names]:
        train_df = pd.concat([dfs[k] for k in train_names]).reset_index(drop=True)
        kappas = []
        for seed in range(10):
            grid = model_selection.GridSearchCV(
                ensemble.RandomForestClassifier(random_state=seed),
                {
                    'min_samples_leaf': [1, 2, 4, 8, 16, 32],
                },
                scoring=metrics.make_scorer(metrics.cohen_kappa_score))
            grid.fit(train_df[features], train_df.label)
            preds = grid.predict(test_df[features])
            kappas.append(metrics.cohen_kappa_score(test_df.label, preds))
        print(train_names, '=>', test_name, 'mean kappa:', pd.Series(kappas).mean())
