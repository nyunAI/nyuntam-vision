import timm
import torch.nn as nn
import os
import sys

sys.path.append(os.path.abspath(os.path.join("..", "utils")))
from vision.core.utils.modelutils import modify_head_classification
from .custom_models import register_custom_timm_models


def get_timm_model(model_name, num_classes, pretrained):
    register_custom_timm_models()
    model = timm.create_model(model_name, pretrained=pretrained)
    modify_head_classification(model, model_name, num_classes)
    return model
