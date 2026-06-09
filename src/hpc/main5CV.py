import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing
import sys
from pathlib import Path
import re
sys.path.insert(0, str(Path(__file__).parent))
from modelClass import base, RF
import argparse


def predictor_columns(data, predictor_prefix, excluded_columns):
    """Select predictor columns without matching response/metadata substrings."""
    prefixes = predictor_prefix.split('|')
    columns = []
    for col in data.columns:
        if col in excluded_columns:
            continue
        for prefix in prefixes:
            if prefix in {"mean", "std"}:
                if re.fullmatch(rf"{re.escape(prefix)}_[0-9]+", col):
                    columns.append(col)
                    break
            elif col.startswith(prefix):
                columns.append(col)
                break
    return columns

parser = argparse.ArgumentParser(description="Train random forest with k-fold grouped cross validation")
parser.add_argument("input_data", type=Path, help="Path to csv with values to predict and predictor values")
parser.add_argument("--k", type=int, default=5, help="Number of folds.")
parser.add_argument('--image_id_col', type=str, default='image_name', help="Image containing unique image IDs for traceability & debugging")
parser.add_argument("--predictor_prefix", type=str, default='feature', help="Substring of predictor column names.")
parser.add_argument("--label", type=str, default='score_average', help="Column containing target labels to predict.")
parser.add_argument("--group", type=str, default='genotype', help="Column containing grouping information.")
parser.add_argument("--output_prefix", type=str, default="output/", help="Prefix for output files.")
args = parser.parse_args()

# Number of folds for CV
k = args.k

# Load dataset
data = pd.read_csv(args.input_data)

data = data.dropna(subset=[args.group, args.label])
predictions_ds = pd.DataFrame()
importance_ds = pd.DataFrame()

kfold = GroupKFold(n_splits = k)
excluded_columns = {args.label, args.group, args.image_id_col}
feature_cols = predictor_columns(data, args.predictor_prefix, excluded_columns)
features = data[feature_cols]
scores = data[args.label]
genotype = data[args.group]
i = 1
    
splits = kfold.split(X = features, y=scores, groups = genotype)
for train_idx, test_idx in splits:
    print('Starting fold: ' + str(i))
    train_ds = data.iloc[train_idx]
    test_ds = data.iloc[test_idx]
        
    train_features = train_ds[feature_cols]
    train_response = train_ds[args.label]

    test_features = test_ds[feature_cols]
    test_images = test_ds[args.image_id_col]    
    test_features = preprocessing.StandardScaler().fit_transform(test_features)
    test_response = test_ds[args.label]
        
    model = RF(response = train_response, features = train_features, rescale_type = 'norm')
    model.grid_search()
    model = model.train_rf(response = train_response, features = train_features)
        
    predictions = model.predict(test_features)
    predictions = predictions.flatten()
        
    fold_predictions = pd.DataFrame({args.image_id_col: test_images,
				     'label': test_response,
                                     'predicted': predictions})
    predictions_ds = pd.concat([predictions_ds, fold_predictions])
        
    importance = model.feature_importances_
    importance = importance.tolist()
    importance = pd.DataFrame(importance).T
        
    importance_ds = pd.concat([importance_ds, importance])
    print('Fold ' + str(i) + ' of ' + str(k) + ' done')
    i = i + 1

# importance_ds = pd.DataFrame.from_dict(importance_ds)        
predictions_ds.to_csv(args.output_prefix + 'predictions_rf.csv')
importance_ds.to_csv(args.output_prefix + 'feature_importances_rf.csv')
print('DONE')
