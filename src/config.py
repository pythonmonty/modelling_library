from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.utils import Bunch
import json
from copy import copy, deepcopy
import os
import logging
import datetime


class ModelConfig:

    def __init__(self,
                 train_conf: Bunch,
                 hyperpar: HyperParamXGB):

        self.data_fn = train_conf.data_fn
        self.features_num = train_conf.features_num
        self.features_cat = train_conf.features_cat
        self.stratify = train_conf.stratify

        self.train_size = train_conf.train_size
        self.val_size = train_conf.val_size
        self.test_size = train_conf.test_size

        self.random_seed = train_conf.random_seed
        self.cat_encoding = train_conf.cat_encoding
        self.model = train_conf.model
        self.clfreg = train_conf.clfreg

        self.hyperparam = hyperpar


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
        self.cv_method = hp.cv_method
        self.cv_set = hp.cv_set
        self.n_folds = hp.n_folds

        self.param_dict = hp.param_dict


def get_config_from_json(json_conf: dict) -> ModelConfig:
    """
    Takes the config as dictionary and initializes the FreqSevConfig class.
    @param json_conf: Dictionary with the config.
    @return: Config as the FreqSevConfig object.
    """
    b = Bunch(**json_conf)
    train_conf = Bunch(**b.training_config)

    hp = HyperParamXGB(b.hyperparams)

    conf = ModelConfig(train_conf=train_conf,
                       hyperpar=hp)

    __read_standard_conf_input(conf, p)
    conf.freq_conf.set_hyperparams(hyperparams)
    conf.sev_conf.set_hyperparams(hyperparams_sev)

    return conf