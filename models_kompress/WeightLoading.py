import os
import torch
import torch.nn as nn
from collections import OrderedDict
def find_state_or_model(loaded):
    if isinstance(loaded, OrderedDict) or  isinstance(loaded, dict):
        return "STATE_DICT"
    else:
        return "MODEL"

def load_model_or_weights(path):
    loaded = torch.load(path)
    if find_state_or_model(loaded) == "MODEL":
        return loaded, 1
    else:
        return loaded, 0
