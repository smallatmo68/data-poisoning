import json
import os
from typing import List


class ConfigManager:
    def __init__(self, model_conf=None, inf_func_conf=None, training_conf=None):
        inf_funcs = self.__read_conf(inf_func_conf)
        model = self.__read_conf(model_conf)
        training = self.__read_conf(training_conf)

        if training:
            self.training_conf = ConfigManager._TrainingEnv(training)
        if model:
            self.model_conf = ConfigManager._Model(model)
        if inf_funcs:
            self.inf_func_conf = ConfigManager._InfluenceFunctions(inf_funcs)

    def __read_conf(self, conf):
        if conf is not None:
            with open(conf, "r") as f:
                conf = json.load(f)
        return conf

    class _TrainingEnv:
        __REG_STRENGTH = "regularization_strength"
        __LEARNING_RATE = "learning_rate"
        __EPOCHS = "epochs"
        __BATCH_SIZE = "batch_size"
        __RANDOM_SEED = "random_seed"

        def __init__(self, data):
            self.data = data

        def get_learning_rate(self):
            return self.data.get(ConfigManager._TrainingEnv.__LEARNING_RATE, 1e-2)

        def model_regularization_strength(self):
            return self.data.get(ConfigManager._TrainingEnv.__REG_STRENGTH, 0)

        def get_epochs(self):
            return self.data.get(ConfigManager._TrainingEnv.__EPOCHS, 100)

        def get_batch_size(self):
            return self.data.get(ConfigManager._TrainingEnv.__BATCH_SIZE, 128)

        def get_random_seed(self):
            return self.data.get(ConfigManager._TrainingEnv.__RANDOM_SEED, None)

    class _Model:
        __NAME = "name"
        __RANDOM_SEED = "random_seed"
        __TRAINABLE_LAYERS = "trainable_layers"
        __CKPT_PATH = "ckpt"

        def __init__(self, data):
            self.data = data

        def get_name(self) -> str:
            return self.data[ConfigManager._Model.__NAME]

        def random_seed(self):
            return self.data.get(ConfigManager._Model.__RANDOM_SEED, None)

        def trainable_layers(self):
            return self.data.get(ConfigManager._Model.__TRAINABLE_LAYERS, None)

        def get_ckpt_path(self):
            return self.data.get(ConfigManager._Model.__CKPT_PATH, None)

    class _InfluenceFunctions:
        __BATCH_SIZE = "batch_size"
        __FAST_CP = "fast_cp"

        def __init__(self, data):
            self.data = data

        def fast_cp(self):
            return self.data.get(ConfigManager._InfluenceFunctions.__FAST_CP, False)

        def batch_size(self):
            return self.data.get(ConfigManager._InfluenceFunctions.__BATCH_SIZE, 4096)
