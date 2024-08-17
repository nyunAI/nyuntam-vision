import logging
import wandb
from nyuntam.algorithm import VisionAlgorithm
from abc import abstractmethod


class NNCFQAT(VisionAlgorithm):
    def __init__(self, model, loaders=None, **kwargs):
        self.kwargs = kwargs
        self.model = model
        self.loaders = loaders
        self.wandb = kwargs.get("wandb", True)
        self.dataset_name = kwargs.get("DATASET_NAME", "CIFAR10")
        self.model_name = kwargs.get("MODEL", "resnet50")
        self.imsize = kwargs.get("insize", "32")
        self.batch_size = kwargs.get("BATCH_SIZE", 8)
        self.model_path = kwargs.get("MODEL_PATH", "")
        self.cache_path = kwargs.get("CACHE_PATH", "")
        self.job_path = kwargs.get("JOB_PATH", "")
        self.logging_path = kwargs.get("LOGGING_PATH", "logs")
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Experiment Arguments: {self.kwargs}")
        self.job_id = kwargs.get("JOB_ID", "1")
        if self.wandb:
            wandb.init(project="Kompress NNCF", name=str(self.job_id))
            wandb.config.update(self.kwargs)
        self.qat = kwargs.get("QAT", False)
        self.platform = kwargs.get("PLATFORM", "mmdet")

    @abstractmethod
    def compress_model(self):
        pass
