import wandb
import logging
from nyuntam.algorithm import VisionAlgorithm
from abc import abstractmethod


class NNCF(VisionAlgorithm):
    def __init__(self, model, loaders=None, **kwargs):
        self.kwargs = kwargs
        self.model = model
        self.loaders = loaders
        self.wandb = kwargs.get("wandb", True)
        self.imsize = kwargs.get("insize", "32")
        self.to_train = kwargs.get("TRAINING", True)
        self.model_name = kwargs.get("MODEL", "resnet50")
        self.imsize = kwargs.get("insize", "32")
        self.model_path = kwargs.get("MODEL_PATH", "models")
        self.logging_path = kwargs.get("LOGGING_PATH", "logs")
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Experiment Arguments: {self.kwargs}")
        self.job_id = kwargs.get("JOB_ID", "1")
        if self.wandb:
            wandb.init(project="Kompress NNCF", name=str(self.job_id))
            wandb.config.update(self.kwargs)

    @abstractmethod
    def compress_model(self):
        pass
