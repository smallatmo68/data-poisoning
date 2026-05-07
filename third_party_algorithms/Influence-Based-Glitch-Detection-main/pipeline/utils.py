import inspect
import os
import sys

import torch

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
from models import dispatcher as model_dispatcher


def load_model(model_config, input_shape, num_classes, ckpt_path=None):
    model_name = model_config.get_name()
    random_state = model_config.random_seed()
    tunable_layers = model_config.trainable_layers()
    model = model_dispatcher[model_name](
        num_classes=num_classes, input_shape=input_shape, random_state=random_state
    )
    model.set_trainable_layers(tunable_layers)
    if ckpt_path is not None:
        state_dict = torch.load(ckpt_path)
        model.load_state_dict(state_dict["model_state_dict"])
    return model
