import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing
from modelClass import base, RF
import argparse
from pathlib import Path

parser = argparse.ArgumentParser(description="Train random forest with k-fold grouped cross validation")
parser.add_argument("values_data", type=Path, help="Path to csv with values to predict (left dataframe).")
parser.add_argument("features_data", type=Path, help="Path to csv with predictor features (right dataframe).")
parser.add_argument("--join_column", type=str, required=True, help="Column name to join dataframes by.")
parser.add_argument("--k", type=int, default=5, help="Number of folds.")
parser.add_argument("--predictor_prefix", type=str, default='feature', help="Substring of predictor column names.")
parser.add_argument("--label", type=str, default='score_average', help="Column containing target labels to predict.")
parser.add_argument("--group", type=str, default='genotype', help="Column containing grouping information.")
parser.add_argument("--output_prefix", type=str, default="output/", help="Prefix for output files.")
args = parser.parse_args()

# Number of folds for CV
k = args.k

# Load datasets
# Left dataframe: values to predict
values_df = pd.read_csv(args.values_data, low_memory=False)
# Right dataframe: predictor features
features_df = pd.read_csv(args.features_data, low_memory=False)

# Perform left join (similar to R's left_join from tidyverse)
data = values_df.merge(features_df, on=args.join_column, how='left')
data = data.dropna(subset=[args.group, args.label])
predictions_ds = pd.DataFrame()
importance_ds = pd.DataFrame()

kfold = GroupKFold(n_splits = k)
features = data[[col for col in data.columns if args.predictor_prefix in col]]
scores = data[args.label]
genotype = data[args.group]
i = 1
    
splits = kfold.split(X = features, y=scores, groups = genotype)
for train_idx, test_idx in splits:
    print('Starting fold: ' + str(i))
    train_ds = data.iloc[train_idx]
    test_ds = data.iloc[test_idx]
        
    train_features = train_ds[[col for col in data.columns if args.predictor_prefix in col]]
    train_response = train_ds[args.label]
        
    test_features = test_ds[[col for col in data.columns if args.predictor_prefix in col]]
        
    test_features = preprocessing.StandardScaler().fit_transform(test_features)
    test_response = test_ds[args.label]
        
    model = RF(response = train_response, features = train_features, rescale_type = 'norm')
    model.grid_search()
    model = model.train_rf(response = train_response, features = train_features)
        
    predictions = model.predict(test_features)
    predictions = predictions.flatten()
        
    fold_predictions = pd.DataFrame({'label': test_response,
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
