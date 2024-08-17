import wandb
import logging
from abc import abstractmethod
from nyuntam.algorithm import VisionAlgorithm


class TensorRT(VisionAlgorithm):
    def __init__(self, model, loaders=None, **kwargs):
        self.kwargs = kwargs
        self.model = model
        self.loaders = loaders
        self.wandb = kwargs.get("wandb", True)
        self.device = kwargs.get("DEVICE", "cuda:0")
        self.model_name = kwargs.get("MODEL", "resnet50")
        self.imsize = kwargs.get("insize", "32")
        self.batch_size = kwargs.get("BATCH_SIZE", 32)
        self.to_train = kwargs.get("TRAINING", True)
        self.logging_path = kwargs.get("LOGGING_PATH", "logs")
        self.framework = kwargs["PLATFORM"]
        self.logger = logging.getLogger(__name__)
        self.model_path = kwargs.get("JOB_PATH", "")
        self.onnx_file_path = f"{self.model_path}/intermediate_onnx.onnx"
        self.trt_file_path = f"{self.model_path}/mds.trt"
        self.logger.info(f"Experiment Arguments: {self.kwargs}")
        self.job_id = kwargs.get("JOB_ID", "1")

        if self.wandb:
            wandb.init(project="Kompress TensorRT", name=str(self.job_id))
            wandb.config.update(self.kwargs)

    @abstractmethod
    def compress_model(self):
        pass
