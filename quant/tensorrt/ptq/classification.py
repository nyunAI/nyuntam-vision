import subprocess
import wandb
import torch
import logging
from vision.core.finetune import train
from .main import TensorRT


class TensorRTClassification(TensorRT):
    def __init__(self, model, loaders=None, **kwargs):
        super().__init__(model, loaders, **kwargs)

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
