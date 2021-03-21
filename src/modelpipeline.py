import pandas as pd
import numpy as np
from imblearn.pipeline import make_pipeline, Pipeline
from sklearn.model_selection import cross_val_score, GridSearchCV, KFold, cross_validate, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import make_scorer, roc_auc_score, mean_squared_error
from sklearn import model_selection
from sklearn.model_selection import GridSearchCV
from sklearn import preprocessing
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multioutput import MultiOutputClassifier
import xgboost as xgb
from .preprocessing import PreprocessPipeline
from .utils import prefix_dict_key, custom_roc_auc_score
import logging


class ModelPipeline:
    """
    Base class for the modelling pipeline
    """

    def __init__(self,
                 prepipe: PreprocessPipeline):
        self.prepipe = prepipe
        self.pipeline = prepipe.pipeline
        self.estimator = None

    # def get_feature_importance(self):
    #     """
    #     Initializes a preprocessing pipeline with PCA to determine the features which explain 80% of the variance.
    #
    #     :return: (list) Feature column names with highest importance according to PCA (from highest to lowest)
    #     """
    #     logging.info('Estimating feature importance with PCA...')
    #     # Remember original feature column names
    #     categorical_features = np.array(self.X_train.select_dtypes(include='category').columns)
    #     numeric_features = np.array(self.X_train.select_dtypes(include='float').columns)
    #
    #     # Initialize preprocessing pipeline with PCA
    #     prepipeline_pca = PreprocessPipeline(X=self.X_train, pca_bool=True).pipeline
    #     prepipeline_pca.fit(self.X_train, self.y_train)
    #
    #     # Get updated feature names after preprocessing transformations
    #     list = prepipeline_pca['preprocessor'].transformers_[1][1]['onehot'].get_feature_names(categorical_features)
    #     feature_names = np.append(numeric_features, list)
    #
    #     # Number of PCA components
    #     n_pcs = prepipeline_pca['pca'].components_.shape[0]
    #     most_important = [np.abs(prepipeline_pca['pca'].components_[i]).argmax() for i in range(n_pcs)]
    #     most_important_names = [feature_names[most_important[i]] for i in range(n_pcs)]
    #
    #     logging.info('Most important features: ' + str(most_important_names))
    #
    #     return most_important_names

    def add_model_to_pipe(self):
        logging.info('Starting multiple model evaluation via cross-validation')
        params = self.prepipe.config.get_hyperparam_dict()
        if self.prepipe.config.model == 'xgb':
            if self.prepipe.config.clfreg == 'clf':
                # estimator = xgb.XGBClassifier(**params)
                if len(self.prepipe.y.unique()) > 2:
                    estimator = MultiOutputClassifier(xgb.XGBClassifier())
                else:
                    estimator = xgb.XGBClassifier()
            else:
                estimator = xgb.XGBRegressor(**params)
        else:
            raise ValueError('Unknown model in pipeline. Please choose /"xgb/" in the config')

        model = [(self.prepipe.config.model, estimator)]

        steps = self.pipeline.steps + model
        self.pipeline = Pipeline(steps=steps)

        return self

    def fit(self):
        # Get the search CV method. This will be empty if it's not set.
        conf = self.prepipe.config
        cv_method = conf.hyperparam.cv_method
        param_dict = conf.get_hyperparam_dict()

        X_train = self.prepipe.X_train
        y_train = self.prepipe.y_train

        if cv_method == "GridSearchCV":
            # Define scoring
            scoring = {}
            if param_dict['eval_metric'][0] == 'auc':
                if len(self.prepipe.y.unique()) > 2:
                    scoring['auc'] = make_scorer(custom_roc_auc_score)
                else:
                    scoring['auc'] = make_scorer(roc_auc_score)
                # scoring = make_scorer(roc_auc_score)
            elif param_dict['eval_metric'][0] == 'rmse':
                scoring['rmse'] = make_scorer(mean_squared_error)
                # scoring = make_scorer(mean_squared_error)
            metric = list(scoring.keys())[0]
            logging.info("Starting grid search cross validation...")
            logging.info("n_estimators: " + str(param_dict["n_estimators"]))

            if conf.hyperparam.cv_set == "kfold":
                n_folds = conf.hyperparam.n_folds
                if conf.stratify is None:
                    mysplit = model_selection.KFold(n_splits=n_folds,
                                                    shuffle=True,
                                                    random_state=conf.random_seed)
                else:
                    mysplit = model_selection.StratifiedKFold(n_splits=n_folds,
                                                              shuffle=True,
                                                              random_state=conf.random_seed)
            else:
                raise ValueError('Only /"kfold/" cross validation is a possible config option. '
                                 'Please modify the config file.')

            param_dict = prefix_dict_key(param_dict,
                                         self.prepipe.config.model + '__' + 'estimator__')

            # Grid search with parallelization
            grid_search = GridSearchCV(estimator=self.pipeline,
                                       param_grid=param_dict,
                                       n_jobs=-1,
                                       cv=mysplit,
                                       scoring=scoring,
                                       return_train_score=False,
                                       refit=metric)

            lb = preprocessing.LabelBinarizer()
            y2 = lb.fit_transform(y_train)

            grid_search.fit(X=X_train, y=y2)

            logging.info("Best params: " + str(grid_search.best_params_))
            self.estimator = grid_search.best_estimator_
            eval_metric = grid_search.best_params_

            # Log the CV results
            logging.info('CV results:')
            cv_df = pd.DataFrame.from_dict(grid_search.cv_results_)
            logging.info('\n' + cv_df.to_string())

        else:
            logging.info("Fitting without grid search...")
            # Fetch the eval_metric from param dict
            eval_metric = param_dict["eval_metric"][0]
            self.pipeline.fit(X=X_train, y=y_train)

        logging.info('Hyperparameter tuning finished...')

        return eval_metric, cv_df
