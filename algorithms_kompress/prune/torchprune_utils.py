import argparse
import math
import os
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import List, Union
import numpy as np
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from ultralytics import YOLO, __version__
from ultralytics.nn.modules import Detect, C2f, Conv, Bottleneck
from ultralytics.nn.tasks import attempt_load_one_weight
from ultralytics.utils.checks import check_pip_update_available
from ultralytics.yolo.engine.trainer import BaseTrainer
from ultralytics.yolo.utils import (
    yaml_load,
    RANK,
    DEFAULT_CFG_DICT,
    DEFAULT_CFG_KEYS,
)
from ultralytics.yolo.utils.checks import check_yaml
from ultralytics.yolo.utils.torch_utils import initialize_weights, de_parallel
import logging
import torch_pruning as tp
import torch.functional as F
LOGGER = logging.getLogger("algorithms_kompress.prune.torchprune")

def forward_timm_vit(self, x):
    """https://github.com/huggingface/pytorch-image-models/blob/054c763fcaa7d241564439ae05fbe919ed85e614/timm/models/vision_transformer.py#L79"""
    B, N, C = x.shape
    qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)
    q, k = self.q_norm(q), self.k_norm(k)

    
    q = q * self.scale
    attn = q @ k.transpose(-2, -1)
    attn = attn.softmax(dim=-1)
    attn = self.attn_drop(attn)
    x = attn @ v

    x = x.transpose(1, 2).reshape(B, N, -1) # original implementation: x = x.transpose(1, 2).reshape(B, N, C)
    x = self.proj(x)
    x = self.proj_drop(x)
    return x
def infer_shortcut(bottleneck):
    c1 = bottleneck.cv1.conv.in_channels
    c2 = bottleneck.cv2.conv.out_channels
    return c1 == c2 and hasattr(bottleneck, "add") and bottleneck.add


class C2f_v2(nn.Module):
    # CSP Bottleneck with 2 convolutions
    def __init__(
        self, c1, c2, n=1, shortcut=False, g=1, e=0.5
    ):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv0 = Conv(c1, self.c, 1, 1)
        self.cv1 = Conv(c1, self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(
            Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0)
            for _ in range(n)
        )

    def forward(self, x):
        # y = list(self.cv1(x).chunk(2, 1))
        y = [self.cv0(x), self.cv1(x)]
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


def transfer_weights(c2f, c2f_v2):
    c2f_v2.cv2 = c2f.cv2
    c2f_v2.m = c2f.m

    state_dict = c2f.state_dict()
    state_dict_v2 = c2f_v2.state_dict()

    # Transfer cv1 weights from C2f to cv0 and cv1 in C2f_v2
    old_weight = state_dict["cv1.conv.weight"]
    half_channels = old_weight.shape[0] // 2
    state_dict_v2["cv0.conv.weight"] = old_weight[:half_channels]
    state_dict_v2["cv1.conv.weight"] = old_weight[half_channels:]

    # Transfer cv1 batchnorm weights and buffers from C2f to cv0 and cv1 in C2f_v2
    for bn_key in ["weight", "bias", "running_mean", "running_var"]:
        old_bn = state_dict[f"cv1.bn.{bn_key}"]
        state_dict_v2[f"cv0.bn.{bn_key}"] = old_bn[:half_channels]
        state_dict_v2[f"cv1.bn.{bn_key}"] = old_bn[half_channels:]

    # Transfer remaining weights and buffers
    for key in state_dict:
        if not key.startswith("cv1."):
            state_dict_v2[key] = state_dict[key]

    # Transfer all non-method attributes
    for attr_name in dir(c2f):
        attr_value = getattr(c2f, attr_name)
        if not callable(attr_value) and "_" not in attr_name:
            setattr(c2f_v2, attr_name, attr_value)

    c2f_v2.load_state_dict(state_dict_v2)


def replace_c2f_with_c2f_v2(module):
    for name, child_module in module.named_children():
        if isinstance(child_module, C2f):
            # Replace C2f with C2f_v2 while preserving its parameters
            shortcut = infer_shortcut(child_module.m[0])
            c2f_v2 = C2f_v2(
                child_module.cv1.conv.in_channels,
                child_module.cv2.conv.out_channels,
                n=len(child_module.m),
                shortcut=shortcut,
                g=child_module.m[0].cv2.conv.groups,
                e=child_module.c / child_module.cv2.conv.out_channels,
            )
            transfer_weights(child_module, c2f_v2)
            setattr(module, name, c2f_v2)
        else:
            replace_c2f_with_c2f_v2(child_module)


def save_model_v2(self: BaseTrainer):
    """
    Disabled half precision saving. originated from ultralytics/yolo/engine/trainer.py
    """
    ckpt = {
        "epoch": self.epoch,
        "best_fitness": self.best_fitness,
        "model": deepcopy(de_parallel(self.model)),
        "ema": deepcopy(self.ema.ema),
        "updates": self.ema.updates,
        "optimizer": self.optimizer.state_dict(),
        "train_args": vars(self.args),  # save as dict
        "date": datetime.now().isoformat(),
        "version": __version__,
    }

    # Save last, best and delete
    torch.save(ckpt, self.last)
    if self.best_fitness == self.fitness:
        torch.save(ckpt, self.best)
    if (
        (self.epoch > 0)
        and (self.save_period > 0)
        and (self.epoch % self.save_period == 0)
    ):
        torch.save(ckpt, self.wdir / f"epoch{self.epoch}.pt")
    del ckpt


def final_eval_v2(self: BaseTrainer):
    """
    originated from ultralytics/yolo/engine/trainer.py
    """
    for f in self.last, self.best:
        if f.exists():
            strip_optimizer_v2(f)  # strip optimizers
            if f is self.best:
                LOGGER.info(f"\nValidating {f}...")
                self.metrics = self.validator(model=f)
                self.metrics.pop("fitness", None)
                self.run_callbacks("on_fit_epoch_end")


def strip_optimizer_v2(f: Union[str, Path] = "best.pt", s: str = "") -> None:
    """
    Disabled half precision saving. originated from ultralytics/yolo/utils/torch_utils.py
    """
    x = torch.load(f, map_location=torch.device("cpu"))
    args = {
        **DEFAULT_CFG_DICT,
        **x["train_args"],
    }  # combine model args with default args, preferring model args
    if x.get("ema"):
        x["model"] = x["ema"]  # replace model with ema
    for k in "optimizer", "ema", "updates":  # keys
        x[k] = None
    for p in x["model"].parameters():
        p.requires_grad = False
    x["train_args"] = {
        k: v for k, v in args.items() if k in DEFAULT_CFG_KEYS
    }  # strip non-default keys
    # x['model'].args = x['train_args']
    torch.save(x, s or f)
    mb = os.path.getsize(s or f) / 1e6  # filesize
    LOGGER.info(
        f"Optimizer stripped from {f},{f' saved as {s},' if s else ''} {mb:.1f}MB"
    )


def train_v2(self: YOLO, pruning=False, trainer=None, **kwargs):
    """
    Trains the model on a given dataset.

    Args:
        trainer (BaseTrainer, optional): Customized trainer.
        **kwargs (Any): Any number of arguments representing the training configuration.
    """
    self._check_is_pytorch_model()
    if self.session:  # Ultralytics HUB session
        if any(kwargs):
            LOGGER.warning(
                "WARNING ⚠️ using HUB training arguments, ignoring local training arguments."
            )
        kwargs = self.session.train_args
    check_pip_update_available()
    overrides = self.overrides.copy()
    if kwargs.get("cfg"):
        LOGGER.info(f"cfg file passed. Overriding default params with {kwargs['cfg']}.")
        overrides = yaml_load(check_yaml(kwargs["cfg"]))
    overrides.update(kwargs)
    overrides["mode"] = "train"
    if not overrides.get("data"):
        raise AttributeError(
            "Dataset required but missing, i.e. pass 'data=coco128.yaml'"
        )
    if overrides.get("resume"):
        overrides["resume"] = self.ckpt_path
    self.task = overrides.get("task") or self.task
    trainer = trainer or self.smart_load("trainer")
    self.trainer = trainer(overrides=overrides, _callbacks=self.callbacks)
    if not pruning:
        if not overrides.get("resume"):  # manually set model only if not resuming
            self.trainer.model = self.trainer.get_model(
                weights=self.model if self.ckpt else None, cfg=self.model.yaml
            )
            self.model = self.trainer.model
    else:
        # pruning mode
        self.trainer.pruning = True
        self.trainer.model = self.model

        # replace some functions to disable half precision saving
        self.trainer.save_model = save_model_v2.__get__(self.trainer)
        self.trainer.final_eval = final_eval_v2.__get__(self.trainer)
    self.trainer.hub_session = self.session  # attach optional HUB session
    self.trainer.train()
    # Update model and cfg after training
    if RANK in (-1, 0):
        self.model, _ = attempt_load_one_weight(str(self.trainer.best))
        self.overrides = self.model.args
        self.metrics = getattr(self.trainer.validator, "metrics", None)