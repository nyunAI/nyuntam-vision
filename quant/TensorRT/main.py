import subprocess
import wandb
import os
import sys
sys.path.append(os.path.abspath(os.path.join("..", "logging_kompress")))
from logging_kompress import define_logger
import torch
from core.finetune import train


class TensorRT:
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
        self.logger = define_logger(
            __name__, self.logging_path
        )
        self.onnx_file_path = f"{self.model_path}/intermediate_onnx.onnx"
        self.trt_file_path = f"{self.model_path}/mds.trt"
        self.logger.info(f"Experiment Arguments: {self.kwargs}")
        self.job_id = kwargs.get("JOB_ID","1")
        if self.wandb:
            wandb.init(project="Kompress TensorRT", name=str(self.job_id))
            wandb.config.update(self.kwargs)
    def convert_to_onnx(self):
        rand_inp = torch.randn(self.batch_size, 3, self.imsize, self.imsize)
        torch.onnx.export(
                self.model.to(self.device), rand_inp.to(self.device), self.onnx_file_path
            )
        return "SUCCESS"
    
    def convert_tensorrt(self):
        subprocess.run(
            f"trtexec --onnx={self.onnx_file_path} --saveEngine={self.trt_file_path} --int8",
            shell=True,
        )

        self.logger.info("TRT Conversion Success")
        return "SUCCESS"

    def compress_model(self):
        if self.to_train:
                self.model, _, _ = train(
                    self.loaders["train"],
                    self.loaders["test"],
                    self.model,
                    __name__,
                    self.kwargs,
                )
        self.model = self.model.to("cpu")
        self.convert_to_onnx()
        self.convert_tensorrt()
        return "None", __name__
