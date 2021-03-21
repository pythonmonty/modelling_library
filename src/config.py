from __future__ import annotations
from sklearn.utils import Bunch
import logging


class ModelConfig:

    def __init__(self,
                 train_conf: Bunch,
                 hyperpar: HyperParamXGB):

        self.data_fn = train_conf.data_fn
        self.target = train_conf.target
        self.features_num = train_conf.features_num
        self.features_cat = train_conf.features_cat
        self.stratify = train_conf.stratify

        if train_conf.resampling is not None and train_conf.resampling in ('smote'):
            self.resampling = train_conf.resampling
        else:
            logging.info('Unknown resampling method. Please choose /"smote/" as config option.')
            self.resampling = None

        if train_conf.train_size == 0 or train_conf.test_size == 0:
            raise ValueError('Cannot train model with train or test size 0. '
                             'Check your configuration.')
        if train_conf.train_size + train_conf.test_size != 1:
            raise ValueError('Train/val/test set sizes do not add up to 1. '
                             'Check your configuration.')
        else:
            self.train_size = train_conf.train_size
            self.test_size = train_conf.test_size

        self.random_seed = train_conf.random_seed
        if train_conf.cat_encoding in ('OneHot', 'Label'):
            self.cat_encoding = train_conf.cat_encoding
        else:
            raise ValueError('Unknown encoding method ' + str(train_conf.cat_encoding))
        if train_conf.model in ('xgb'):
            self.model = train_conf.model
        else:
            raise ValueError('Unknown model ' + str(train_conf.model))
        if train_conf.clfreg in ('clf', 'reg'):
            self.clfreg = train_conf.clfreg
        else:
            raise ValueError('Only classification (clf) and regression (reg) are possible. '
                             'Unknown method ' + str(train_conf.clfreg))

        self.hyperparam = hyperpar

    def get_hyperparam_dict(self):
        return self.hyperparam.param_dict


class HyperParamXGB:
    """
    Class which holds the hyperparameters of the XGB models.

    Parameters:
        @ivar cv_method: If using hyperparameter tuning, should be filled with 'GridSearchCV'
        @ivar cv_set: If using hyperparameter tuning, should be filled with 'kfold'
        @ivar n_folds: If using hyperparameter tuning, choose number of cross-validation folds
        @ivar param_dict: Dictionary holding XGB hyperparameters
    """

    def __init__(self, hp: Bunch):
        if hp.cv_method in ('GridSearchCV'):
            self.cv_method = hp.cv_method
        else:
            raise ValueError('Unknown cross-validation method ' + str(hp.cv_method))
        if hp.cv_set in ('kfold', 'single'):
            self.cv_set = hp.cv_set
        else:
            raise ValueError('Unknown cross-validation set ' + str(hp.cv_set))
        if isinstance(hp.n_folds, int):
            self.n_folds = hp.n_folds
        else:
            raise ValueError('Wrong type ' + str(type(hp.n_folds)) + ' for ' + str(hp.n_folds))

        # TODO: add sanity checks for hyperparameters of XGB
        self.param_dict = hp.param_dict


def get_config_from_json(json_conf: dict) -> ModelConfig:
    """
    Takes the config as dictionary and initializes the FreqSevConfig class.
    @param json_conf: Dictionary with the config.
    @return: Config as the FreqSevConfig object.
    """
    b = Bunch(**json_conf)
    train_conf = Bunch(**b.training_config)

    hp = HyperParamXGB(Bunch(**b.hyperparam))

    conf = ModelConfig(train_conf=train_conf,
                       hyperpar=hp)

    return conf
