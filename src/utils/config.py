# -*- coding: utf-8 -*-
"""Config class"""

import json


class Config:
    """Config class which contains data, train and model hyperparameters"""

    def __init__(
        self, data, dataloaders, train_setup, tuning, model_params, mlflow_config
    ):
        self.data = data
        self.dataloaders = dataloaders
        self.train_setup = train_setup
        self.tuning = tuning
        self.model_params = model_params
        self.mlflow_config = mlflow_config

    @classmethod
    def from_json(cls, cfg):
        """Creates config from json"""
        params = json.loads(json.dumps(cfg), object_hook=HelperObject)
        return cls(
            params.data,
            params.dataloaders,
            params.train_setup,
            params.tuning,
            params.model_params,
            params.mlflow_config,
        )


class HelperObject(object):
    """Helper class to convert json into Python object"""

    def __init__(self, dict_):
        self.__dict__.update(dict_)
