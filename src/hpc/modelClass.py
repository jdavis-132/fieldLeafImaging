import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import (cross_val_score,
    RepeatedKFold,
    RandomizedSearchCV,
    KFold,
    train_test_split,
    GridSearchCV,
    GroupKFold,
    train_test_split,GroupShuffleSplit)                                
from sklearn.metrics import mean_squared_error
# from sklearn.exceptions import ConvergenceWarning
# Suppress ConstantInputWarning
# warnings.filterwarnings("ignore", category=stats.ConstantInputWarning)

# warnings.filterwarnings("ignore", category=ConvergenceWarning)

class base:
    def __init__(self, **kwargs):
        self.features = kwargs["features"]
        self.response = kwargs["response"]

    def preProcess_features(self, **kwargs):
        if kwargs["rescale_type"] == "norm":
            standardized = preprocessing.StandardScaler()
            features_processed = standardized.fit_transform(np.array(self.features))
        elif kwargs["rescale_type"] == "minmax":
            norm = preprocessing.MinMaxScaler()
            features_processed = norm.fit_transform(np.array(self.features))
        else:
            print(str(kwargs["rescale_type"])+ "rescaling technique not implemented \n Defaulting to standardized variables")
            standardized = preprocessing.StandardScaler()
            features_processed = standardized.fit_transform(np.array(self.features))

        self.features_processed = features_processed
        return features_processed

    def predict(self, **kwargs):
        features_to_predict = kwargs["features"]
        self.predictions = self.model.predict(features_to_predict)

    def run_CVs(self, **kwargs):

        standardized = preprocessing.StandardScaler()
        norm = preprocessing.MinMaxScaler()

        x_data = kwargs["features"]
        rescale_type = kwargs['rescale_type']

        # Normalise or standardize, two different forms of rescaling
        if kwargs["rescale_type"] == "norm":
            x_data = standardized.fit_transform(np.array(x_data))
        elif kwargs["rescale_type"] == "minmax":
            x_data = norm.fit_transform(np.array(x_data))
        else:
            print("rescaling technique not implemented \n Defaulting to standardized variables")
            x_data = standardized.fit_transform(np.array(x_data))

        y_data = np.array(kwargs["response"])
        n_folds = kwargs["n_folds"]
        title = kwargs["title"]

        kf = KFold(n_splits=n_folds)
        df = {}
        fold_indices = {}

        count = 1
        for train_index, test_index in kf.split(x_data):
            X_train, X_test = x_data[train_index], x_data[test_index]
            y_train, y_test = y_data[train_index], y_data[test_index]

            model_cv = kwargs["model_type"]
            model_cv.fit(X_train, y_train)

            fold = model_cv.predict(X_test)
            df[f"Pred{count}"] = fold
            df[f"Obs{count}"] = y_test

            fold_indices[f"Train{count}"] = train_index
            fold_indices[f"Test{count}"] = test_index

            count += 1

        fig = plt.figure(figsize=(20, 15))

        count = 1

        comb_cv_obs = []
        comb_cv_preds = []

        # Wether or not to visualise the cross validation
        try:
            visualize = kwargs["visualize"]
        except:
            visualize = None

        if visualize:
            for i in range(n_folds):
                print(i)
                print(f"23{count}")
                ax = fig.add_subplot(int(n_folds/2),2,count)
                sns.regplot(x=df[f"Obs{count}"], y=df[f"Pred{count}"])
                ax.spines["right"].set_visible(False)
                ax.spines["top"].set_visible(False)
                ax.set_ylabel("Predicted")
                ax.set_xlabel("Observed")
                ax.set_title(f"Fold{count}")
                r_val, pval = spearmanr(df[f"Obs{count}"], df[f"Pred{count}"])
                r2_val = round(r_val ** 2, 2)
                x_cord, y_cord = max(df[f"Obs{count}"]) * 0.15, max(df[f"Pred{count}"])
                ax.annotate(f"$R^2 = {r2_val}$", (x_cord, y_cord))

                comb_cv_preds.extend(df[f"Pred{count}"])
                comb_cv_obs.extend(df[f"Obs{count}"])

                count += 1

            R, pVal = spearmanr(comb_cv_obs, comb_cv_preds)
            R2 = round(R ** 2, 2)

            fig.suptitle(f"{title} (Combined data $R^2$ = {R2})")
            plt.tight_layout()
            direc = os.getcwd()
            out_direc = f"{direc}"
            os.makedirs(out_direc, exist_ok=True)
            # plt.savefig(f'{out_direc}/Fold_{title}.png')
            # plt.savefig(f'{out_direc}/Fold_{title}.svg')
            # plt.show()

            fig2 = plt.figure(figsize=(12, 10))
            ax2 = fig2.add_subplot(111)
            sns.regplot(x=comb_cv_obs, y=comb_cv_preds, ax=ax2)
            ax2.spines["top"].set_visible(False)
            ax2.spines["right"].set_visible(False)
            ax2.set_title(title, pad=10)
            r, pval = spearmanr(comb_cv_obs, comb_cv_preds)
            r2 = round(r ** 2, 2)
            x_coord = max(comb_cv_obs) * 0.75
            y_coord = max(comb_cv_preds) * 0.99
            ax2.text(x_coord, y_coord, f"$R^2 = {round(r2,2)}$")
            ax2.set_xlabel("Observations", labelpad=20)
            ax2.set_ylabel("Predictions", labelpad=20)
            fig2.tight_layout()
            fig2.savefig(f'{out_direc}/{title}.png') 
            # fig2.savefig(f'{out_direc}/{title}.svg')

        return df, fold_indices
    
    
    import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import (cross_val_score,
    RepeatedKFold,
    RandomizedSearchCV,
    KFold,
    train_test_split,
    GridSearchCV,
    GroupKFold,
    train_test_split,GroupShuffleSplit)                                
from sklearn.metrics import mean_squared_error
# from sklearn.exceptions import ConvergenceWarning
# Suppress ConstantInputWarning
# warnings.filterwarnings("ignore", category=stats.ConstantInputWarning)
# warnings.filterwarnings("ignore", category=ConvergenceWarning)
# import base

class RF(base):

    # """" Random Forest Regression Model """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.features_processed = self.preProcess_features(rescale_type=kwargs.get("rescale_type"))


    def grid_search(self, **kwargs):
        n_estimators = [int(x) for x in np.linspace(start=100, stop=1000, num=10)]
        max_features = [1, 2, 3, 4,5,6,7,8,9,10]
        max_depth = [int(x) for x in np.linspace(10, stop=100, num=11)]
        max_depth.append(None)
        min_samples_split = [2, 5, 10]
        min_samples_leaf = [10, 15, 20]
        bootstrap = [True, False]
        random_grid = {
            "n_estimators": n_estimators,
            "max_features": max_features,
            "max_depth": max_depth,
            "min_samples_split": min_samples_split,
            "min_samples_leaf": min_samples_leaf,
            "bootstrap": bootstrap}
       
        rf = RandomForestRegressor()
       
        rf_random = RandomizedSearchCV(
            estimator=rf,
            param_distributions=random_grid,
            n_iter=100,
            cv=3,
            verbose=0,
            random_state=42,
            n_jobs=-1)
        rf_random.fit(self.features_processed, self.response)
        # print(rf_random.best_params_)

        self.ran_params = rf_random.best_params_

    def train_rf(self, **kwargs):
        model = RandomForestRegressor(
            n_estimators=self.ran_params["n_estimators"],
            min_samples_leaf=self.ran_params["min_samples_leaf"],
            min_samples_split=self.ran_params["min_samples_split"],
            max_features=self.ran_params["max_features"],
            max_depth=self.ran_params["max_depth"],
            bootstrap=self.ran_params["bootstrap"])

        model.fit(self.features_processed, self.response)
        self.model = model
        
        return model

    def run_CVs(self, **kwargs):
        df, fold_indices = super().run_CVs(
            model_type=RandomForestRegressor(
                n_estimators=self.ran_params["n_estimators"],
                min_samples_leaf=self.ran_params["min_samples_leaf"],
                min_samples_split=self.ran_params["min_samples_split"],
                max_features=self.ran_params["max_features"],
                max_depth=self.ran_params["max_depth"],
                bootstrap=self.ran_params["bootstrap"]),
            features=kwargs["features"],
            response=kwargs["response"],
            n_folds=kwargs["n_folds"],
            title=kwargs["title"],
            visualize=kwargs["visualize"],
            rescale_type=kwargs["rescale_type"])

        return df, fold_indices

    def feature_importances(self, **kwargs):

        importances = self.model.feature_importances_
        self.feature_importance_std = np.std([tree.feature_importances_ for tree in self.model.estimators_], axis=0)
        return importances