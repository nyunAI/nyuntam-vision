import os

# Create a folder for downloaded checkpoints in the user folder
import subprocess
from glob import glob
from mmdet.utils import setup_cache_size_limit_of_dynamo
from mmengine.config import Config
import shutil
from mmengine.registry import RUNNERS
from mmengine.runner import Runner


def get_mmpose_model(name, kwargs):
    model_name = name
    cache_path = kwargs.get("CACHE_PATH", "")
    launcher = kwargs.get("LAUNCHER", "none")
    intermediate_path = os.path.join(cache_path, f"intermediate_{model_name}")
    amp = kwargs.get("AMP", False)
    auto_scale_lr = kwargs.get("AUTO_SCALE_LR", False)
    if not os.path.exists(intermediate_path):
        os.makedirs(intermediate_path)
        # Download the weights there
        cpath = os.getcwd()
        os.chdir(intermediate_path)
        os.system(f"mim download mmpose --config {model_name} --dest .")
        os.chdir(cpath)

    setup_cache_size_limit_of_dynamo()
    cfg_path = glob(f"{intermediate_path}/*.py")[0]
    cfg = Config.fromfile(cfg_path)
    cfg.launcher = launcher

    changes = {"data_root": kwargs.get("DATA_PATH", "")}
    cfg.merge_from_dict(changes)
    cfg.work_dir = os.path.join(cache_path)
    if amp is True:
        cfg.optim_wrapper.type = "AmpOptimWrapper"
        cfg.optim_wrapper.loss_scale = "dynamic"

    if auto_scale_lr:
        if (
            "auto_scale_lr" in cfg
            and "enable" in cfg.auto_scale_lr
            and "base_batch_size" in cfg.auto_scale_lr
        ):
            cfg.auto_scale_lr.enable = True
        else:
            raise RuntimeError(
                'Can not find "auto_scale_lr" or '
                '"auto_scale_lr.enable" or '
                '"auto_scale_lr.base_batch_size" in your'
                " configuration file."
            )
    # Load the runner
    if "runner_type" not in cfg:
        # build the default runner
        runner = Runner.from_cfg(cfg)
    else:
        # build customized runner from the registry
        # if 'runner_type' is set in the cfg
        runner = RUNNERS.build(cfg.runner)
    return runner
