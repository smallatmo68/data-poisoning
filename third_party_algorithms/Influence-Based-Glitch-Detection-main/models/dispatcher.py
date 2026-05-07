from .pretrained.convnext import ConvNextModel
from .pretrained.resnet20 import Resnet20Model
from .pretrained.vit import VisionTransformerModel

dispatcher = {
    "resnet20": Resnet20Model,
    "vit": VisionTransformerModel,
    "convnext": ConvNextModel,
}
