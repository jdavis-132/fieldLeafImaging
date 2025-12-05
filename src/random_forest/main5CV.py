import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing
from modelClass import base, RF
# from pathlib import Path

# Number of folds for CV
k = 5

# Load your dataset
# Assuming you have a DataFrame named 'data' with columns: plantDensity, kernelRowNumber, kernelsPerRow, hundredKernelMass, yieldPerAcre
# Adjust the file path or loading mechanism as needed
data = pd.read_csv("output/dinov2_features_metadata.csv")

predictions_ds = pd.DataFrame())
importance_ds = pd.DataFrame()

kfold = GroupKFold(n_splits = k)
features = data[[col for col in df.columns if 'feature' in col]]
scores = data['score_average']
genotype = data['genotype']
    
splits = kfold.split(X = features, y=scores, groups = genotype)
for train_idx, test_idx in splits:
    train_ds = data.iloc[train_idx]
    test_ds = data.iloc[test_idx]
        
    train_features = train_ds[['plantDensity.sp', 'kernelRowNumber.sp', 'kernelsPerRow.sp', 'hundredKernelMass.sp']]
    train_response = train_ds['yieldPerAcre.sp']
        
    test_features = test_ds[['plantDensity.sp', 'kernelRowNumber.sp', 'kernelsPerRow.sp', 'hundredKernelMass.sp']]
        
    test_features = preprocessing.StandardScaler().fit_transform(test_features)
    test_response = test_ds['yieldPerAcre.sp']
    test_plotNumbers = test_ds['plotNumber']
    test_environments = test_ds['environment']
        
    model = RF(response = train_response, features = train_features, rescale_type = 'norm')
    model.grid_search()
    model = model.train_rf(response = train_response, features = train_features)
        
    predictions = model.predict(test_features)
    predictions = predictions.flatten()
        
    fold_predictions = pd.DataFrame({'environment': test_environments,
                                     'plotNumber': test_plotNumbers,
                                     'predictedYield': predictions})
    predictions_ds = pd.concat([predictions_ds, fold_predictions])
        
    importance = model.feature_importances_
    importance = importance.tolist()
    importance = pd.DataFrame(importance).T
        
    importance_ds = pd.concat([importance_ds, importance])

# importance_ds = pd.DataFrame.from_dict(importance_ds)        
predictions_ds.to_csv('../analysis/RFpredictions5CV.csv')
importance_ds.to_csv('../analysis/featureImportances5CV.csv')
print('DONE')
