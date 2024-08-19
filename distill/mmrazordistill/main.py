import wandb
import logging
import os
from glob import glob
import torch
from mmengine.runner import Runner
from .utils import *
from vision.core.utils.modelutils import download_mim_config
from abc import ABC, abstractmethod
from nyuntam.algorithm import VisionAlgorithm

class MMRazorDistill(VisionAlgorithm):
    def __init__(self, teacher_model, model, loaders=None, **kwargs):
        self.kwargs = kwargs
        self.model = model
        self.teacher_model = teacher_model
        self.loaders = loaders
        self.wandb = kwargs.get("wandb", True)
        self.task = kwargs.get("TASK", "image_classification")
        self.device = kwargs.get("DEVICE", "cuda:0")
        self.dataset_name = kwargs.get("DATASET_NAME", "CIFAR10")
        self.model_name = kwargs.get("MODEL", "resnet50")
        self.imsize = kwargs.get("insize", "32")
        self.batch_size = kwargs.get("BATCH_SIZE", 32)
        self.to_train = kwargs.get("TRAINING", True)
        self.folder_name = kwargs.get("USER_FOLDER", "abc")
        self.model_path = kwargs.get("MODEL_PATH", "models")
        self.logging_path = kwargs.get("LOGGING_PATH", "logs")
        self.data_path = kwargs.get("DATA_PATH", "data")
        self.cache_path = kwargs.get("CACHE PATH", ".cache")
        self.custom_teacher_path = kwargs.get("CUSTOM_TEACHER_PATH")
        self.epochs = kwargs.get("EPOCHS", 10)
        self.validation_interval = kwargs.get("VALIDATION_INTERVAL", 1)
        self.logger = logging.getLogger(__name__)
        self.palettes = kwargs.get("PALETTES", [])
        self.classes_list = tuple(kwargs.get("CLASSES_LIST", ()))
        self.teacher_name = kwargs.get("TEACHER_MODEL", "")
        self.logger.info(f"Experiment Arguments: {self.kwargs}")
        self.job_id = kwargs.get("JOB_ID", "1")
        self.platform = kwargs.get("PLATFORM", "torchvision")
        self.insize = kwargs.get("insize", 32)
        self.input_shape = (1, 3, self.insize, self.insize)
        self.epochs = kwargs.get("EPOCHS", 1)
        cache_path = kwargs.get("CACHE_PATH", "")
        self.method = kwargs.get("METHOD", "cwd")
        intermediate_path = os.path.join(cache_path, f"intermediate_{self.model_name}")
        intermediate_teacher_path = os.path.join(
            cache_path, f"intermediate_{self.teacher_name}"
        )
        download_mim_config(self.platform, self.model_name, intermediate_path)
        download_mim_config(self.platform, self.teacher_name, intermediate_teacher_path)
        self.student_config = f"{self.model_name}.py"
        self.teacher_config = f"{self.teacher_name}.py"
        if self.custom_teacher_path != "":
            self.teacher_pth = self.custom_teacher_path
        else:
            self.teacher_pth = glob(f"{intermediate_teacher_path}/*.pth")[0]
        self.logger.info(f"Student Config:{self.student_config}")
        self.logger.info(f"Teacher Config:{self.teacher_config}")
        self.logger.info(f"Teacher Pth:{self.teacher_pth}")
        if self.wandb:
            wandb.init(project="Kompress MMRazor", name=str(self.job_id))
            wandb.config.update(self.kwargs)

    @abstractmethod
    def write_config(self):
        pass

    @abstractmethod
    def customize_config(self):
        pass

    def save_model(self, pruned_model):
        torch.save(pruned_model, f"{self.model_path}/mds.pt")
        if not os.path.exists(f"{self.model_path}/mds.pt"):
            raise Exception("Model Saving Unsuccessful")

    def compress_model(self):
        cfg = self.customize_config()
        self.model.cfg["model"]["_scope_"] = self.platform
        self.teacher_model.cfg["model"]["_scope_"] = self.platform
        cfg["model"]["architecture"] = self.model.cfg["model"]
        cfg["model"]["teacher"] = self.teacher_model.cfg["model"]
        cfg["train_cfg"]["max_epochs"] = self.epochs
        runner = Runner.from_cfg(cfg)
        runner.train()
        return "None", __name__
