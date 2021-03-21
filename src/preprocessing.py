import numpy as np
import pandas as pd
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from .config import ModelConfig
from .utils import generate_dtype_dict
import logging


class PreprocessPipeline:
    """
    Base class to run the preprocessing pipeline including:
    - Imputer for Nan values (imputes mean for numerical and most_frequent for categorical variables)
    - StandardScaler for numerical variables
    - LabelEncoder or OneHotEncoder for categorical non-binary variables
    - SMOTE method to balance out the target class for training (optional)
    """

    def __init__(self,
                 conf: ModelConfig):
        self.config = conf
        logging.info('Initializing preprocessing pipeline...')

        logging.info('Reading input data as Pandas dataframe and setting dtypes for features')
        dtype_dict = generate_dtype_dict(features_cat=self.config.features_cat,
                                         features_num=self.config.features_num)
        self.df_raw = pd.read_csv(self.config.data_fn, sep=';', dtype=dtype_dict)
        self.X = self.df_raw[self.config.features_cat + self.config.features_num]
        self.y = self.df_raw[self.config.target]

        self.X_train = pd.DataFrame()
        self.X_test = pd.DataFrame()
        self.y_train = pd.DataFrame()
        self.y_test = pd.DataFrame()

        self.pipeline = None

    def split_data(self):
        if self.config.stratify is not None:
            y = self.y
        else:
            y = None
        self.X_train, self.X_test, self.y_train, self.y_test = \
            train_test_split(self.X,
                             self.y,
                             test_size=self.config.test_size,
                             train_size=self.config.train_size,
                             random_state=self.config.random_seed,
                             stratify=y)
        logging.info('Divided data into train/test sets')

        return self.X_train, self.X_test, self.y_train, self.y_test

    def set_pipeline(self,
                     imputer_num: str = 'median',
                     imputer_cat: str = 'most_frequent',
                     scale: bool = False):

        # Steps for the transformation of numeric features
        steps_num = [('imputer', SimpleImputer(strategy=imputer_num))]
        logging.info('NaN imputation with ' + str(imputer_num) +
                     ' added to the preprocessing pipeline of numeric features')
        if scale:
            steps_num.append(('scaler', StandardScaler()))
            logging.info('StandardScaler added to the preprocessing pipeline of numeric features')
        numeric_transformer = Pipeline(steps=steps_num)

        # Steps for the transformation of categorical features
        steps_cat = [('imputer', SimpleImputer(strategy=imputer_cat))]
        logging.info('NaN imputation with ' + str(imputer_cat) +
                     ' added to the preprocessing pipeline of categorical features')
        if self.config.cat_encoding == 'OneHot':
            steps_cat.append(('onehot', OneHotEncoder(handle_unknown='ignore')))
            logging.info('OneHotEncoder added to the preprocessing pipeline '
                         'of categorical features')
        else:
            # steps_cat.append(('label', LabelEncoder()))
            steps_cat.append(('label', OrdinalEncoder()))
            logging.info('LabelEncoder added to the preprocessing pipeline '
                         'of categorical features')
        categorical_transformer = Pipeline(steps=steps_cat)

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, self.config.features_num),
                ('cat', categorical_transformer, self.config.features_cat)])

        # Steps of the pipeline
        steps = [('preprocessor', preprocessor)]

        if self.config.resampling is not None:
            # Let us define the oversampling part of the pipeline to balance out the target class in training
            over = SMOTE()
            steps.append(('over', over))
            logging.info('Class balancer SMOTE added to the pipeline')

        self.pipeline = Pipeline(steps=steps)
        logging.info('Preprocessing pipeline is initialized...')

        return self
