from typing import List

import torch
from torch import nn
from torch.nn import Conv2d
from torchvision import models
from torchvision.transforms import Resize

from ..model_functions import ModelFunctions


class VisionTransformerModel(ModelFunctions, nn.Module):
    def __init__(self, num_classes: int, input_shape: tuple = None, random_state=None):
        super(VisionTransformerModel, self).__init__()
        model = models.vit_b_16(pretrained=True)

        self.model = model
        self.num_classes = num_classes
        self.random_state = random_state
        self.__trainable_layers = None
        self.freeze_layers()
        self.set_classification_layer()

    def forward(self, x):
        channels = x.shape[1]
        if channels < 3:
            x = torch.cat([x, x, x], dim=1)
        width = max(x.shape[2], 224)
        height = max(x.shape[3], 224)
        x = Resize((width, height))(x)
        return self.model(x)

    def freeze_layers(self):
        for param in self.model.parameters():
            param.requires_grad = False

    def set_trainable_layers(self, layers: List[str]):
        for name, param in self.model.named_parameters():
            for layer in layers:
                if layer in name:
                    param.requires_grad = True
        if self.__trainable_layers is None:
            self.__trainable_layers = []
        self.__trainable_layers = layers + self.__trainable_layers

    def set_classification_layer(self):
        num_ftrs = self.model.heads.head.in_features
        if self.random_state is not None:
            torch.manual_seed(self.random_state)
        self.model.heads.head = nn.Linear(num_ftrs, self.num_classes)
        if self.__trainable_layers is None:
            self.__trainable_layers = []
        self.__trainable_layers.append("heads")

    def count_params(self, trainable=True):
        params = self.model.parameters()
        if trainable:
            params = filter(lambda p: p.requires_grad, params)
        total_params = sum(p.numel() for p in params)
        return total_params

    def is_pretrained(self):
        return True

    def trainable_layer_names(self):
        layers = []
        for layer, _ in self.named_modules():
            for tl in self.__trainable_layers:
                if layer.endswith(tl):
                    layers.append(layer)
        return list(set(layers))

    def get_model_instance(self):
        return self.model
