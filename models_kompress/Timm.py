import timm
import torch.nn as nn
import os
import sys
sys.path.append(os.path.abspath(os.path.join("..", "utils")))
from utils import modify_head_classification



def get_timm_model(model_name, num_classes, pretrained):
    model = timm.create_model(model_name, pretrained=pretrained)
    modify_head_classification(model, model_name,num_classes)
    return model

