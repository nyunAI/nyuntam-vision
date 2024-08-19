import wandb
import logging
import os
from mmengine.config import Config
from glob import glob
import torch
from mmengine.runner import Runner
from .utils import *
from .main import MMRazorDistill


class MMRazorDistillPoseEstimation(MMRazorDistill):
    def __init__(self, teacher_model, model, loaders=None, **kwargs):
        super().__init__(teacher_model, model, loaders, **kwargs)

    def write_config(self):
        if self.method == "pkd_pose":
            config = return_pkd_pose_config(
                self.teacher_config, self.teacher_pth, self.student_config
            )
        else:
            raise ValueError(
                f"Required Distillation Algorithm {self.method} is not supported"
            )
        with open(f"{self.cache_path}/current_distill_config.py", "w") as f:
            f.write(config)
        return

    def customize_config(self):
        self.write_config()
        cfg = Config.fromfile(f"{self.cache_path}/current_distill_config.py")
        cfg["data_root"] = self.data_path
        if isinstance(self.model, str) and self.model[:4] == "rtmo":
            cfg["dataset_coco"]["data_root"] = self.data_path
            cfg["dataset_coco"]["ann_file"] = os.path.join(
                "root/annotations/person_keypoints_train2017.json"
            )
            cfg["dataset_coco"]["data_prefix"]["img"] = os.path.join("root/train2017/")
        cfg["train_dataloader"]["dataset"]["data_root"] = self.data_path
        cfg["train_dataloader"]["dataset"]["ann_file"] = os.path.join(
            "root/annotations/person_keypoints_train2017.json"
        )
        if "img" in cfg["train_dataloader"]["dataset"]["data_prefix"]:
            cfg["train_dataloader"]["dataset"]["data_prefix"]["img"] = os.path.join(
                "root/train2017/"
            )
        else:
            cfg["train_dataloader"]["dataset"]["data_prefix"] = os.path.join(
                "root/train2017/"
            )
        cfg["val_dataloader"]["dataset"]["data_root"] = self.data_path
        cfg["val_dataloader"]["dataset"]["ann_file"] = os.path.join(
            "root/annotations/person_keypoints_val2017.json"
        )
        if "img" in cfg["val_dataloader"]["dataset"]["data_prefix"]:
            cfg["val_dataloader"]["dataset"]["data_prefix"]["img"] = os.path.join(
                "root/val2017/"
            )
        else:
            cfg["val_dataloader"]["dataset"]["data_prefix"] = os.path.join(
                "root/val2017/"
            )
        cfg["test_dataloader"]["dataset"]["data_root"] = self.data_path
        cfg["test_dataloader"]["dataset"]["ann_file"] = os.path.join(
            "root/annotations/person_keypoints_val2017.json"
        )
        if "img" in cfg["test_dataloader"]["dataset"]["data_prefix"]:
            cfg["test_dataloader"]["dataset"]["data_prefix"]["img"] = os.path.join(
                "root/val2017/"
            )
        else:
            cfg["test_dataloader"]["dataset"]["data_prefix"] = os.path.join(
                "root/val2017/"
            )
        cfg["val_evaluator"]["ann_file"] = os.path.join(
            self.data_path, "root/annotations/person_keypoints_val2017.json"
        )
        cfg["test_evaluator"]["ann_file"] = os.path.join(
            self.data_path, "root/annotations/person_keypoints_val2017.json"
        )
        cfg["train_dataloader"]["batch_size"] = self.batch_size
        cfg["val_dataloader"]["batch_size"] = self.batch_size
        cfg["test_dataloader"]["batch_size"] = self.batch_size
        cfg["input_shape"] = self.input_shape
        cfg["train_cfg"]["max_epochs"] = self.epochs
        cfg.work_dir = self.cache_path
        cfg.dump(f"{self.cache_path}/current_config_new.py")
        return cfg
