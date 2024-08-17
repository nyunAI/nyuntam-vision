import wandb
import os
import sys

sys.path.append(os.path.abspath(os.path.join("...", "core")))
import logging
import numpy as np
from onnxruntime.quantization import (
    quantize_static,
    CalibrationDataReader,
)
import torch
import onnx
import sys
import os
from vision.core.finetune import train
from nyuntam.algorithm import VisionAlgorithm


class ONNXQuant(VisionAlgorithm):
    def __init__(self, model, loaders=None, **kwargs):
        self.kwargs = kwargs
        self.model = model
        self.loaders = loaders
        self.framework = kwargs.get("PLATFORM", "torchvision")
        self.wandb = kwargs.get("wandb", True)
        self.task = kwargs.get("TASK", "image_classification")
        self.device = kwargs.get("DEVICE", "cuda:0")
        self.imsize = kwargs.get("insize", "32")
        self.to_train = kwargs.get("TRAINING", True)
        self.quant_format = kwargs.get("quant_format", "QuantFormat.QDQ")
        self.per_channel = kwargs.get("per_channel", False)
        self.activation_type = kwargs.get("activation_type", "QuantType.QInt8")
        self.weight_type = kwargs.get("weight_type", "QuantType.QInt8")
        self.model_path = kwargs.get("MODEL_PATH", "models")
        self.logging_path = kwargs.get("LOGGING_PATH", "logs")
        self.num_samples = kwargs.get("NUM_SAMPLES", 10)
        self.logger = logging.getLogger(__name__)
        self.onnx_file_path = f"{self.model_path}/ori.onnx"
        self.onnx_quantized_file_path = f"{self.model_path}/mds.onnx"
        self.logger.info(f"Experiment Arguments: {self.kwargs}")
        self.job_id = kwargs.get("JOB_ID", "1")
        if self.wandb:
            wandb.init(project="Kompress ONNXQuant", name=str(self.job_id))
            wandb.config.update(self.kwargs)

        # if self.framework == "ultralytics":
        #     self.model = model.model

    def convert_to_onnx(self):
        rand_inp = torch.randn(self.batch_size, 3, self.imsize, self.imsize)
        torch.onnx.export(
            self.model.to(self.device), rand_inp.to(self.device), self.onnx_file_path
        )
        return "SUCESS"

    def quantize(self):
        if self.to_train:
            model, _, _ = train(
                self.loaders["train"],
                self.loaders["test"],
                self.model,
                __name__,
                self.kwargs,
            )
            self.model = model
            self.model = self.model.to("cpu")

        self.convert_to_onnx()
        dr = DummyDataReader(
            self.num_samples, self.loaders["test"], self.onnx_file_path, self.kwargs
        )

        if not os.path.exists(self.onnx_quantized_file_path):

            quantize_static(
                self.onnx_file_path,
                self.onnx_quantized_file_path,
                dr,
                quant_format=eval(self.quant_format),
                per_channel=self.per_channel,
                activation_type=eval(self.activation_type),
                weight_type=eval(self.weight_type),
            )
        return

    def compress_model(self):
        self.quantize()
        return "None", __name__


class DummyDataReader(CalibrationDataReader):
    def __init__(self, num_samples, dataloader, onnx_path, kwargs):
        self.num_samples = num_samples
        self.current_sample = 0
        self.dataloader = dataloader
        self.framework = kwargs["PLATFORM"]
        self.onnx_path = onnx_path
        self.input_name = self.get_input_name()

    def get_input_name(self):
        model = onnx.load(self.onnx_path)
        input_names = [n.name for n in model.graph.input]
        assert (
            len(input_names) == 1
        ), f"Currently supports only models with singular inputs"
        return input_names[0]

    def get_next(self):
        if self.current_sample < self.num_samples:
            input_data, _ = next(iter(self.dataloader))
            input_data = input_data.numpy()
            self.current_sample += 1
            return {self.input_name: input_data}
        else:
            return None

    def generate_random_input(self):
        input_data = np.random.uniform(-1, 1, size=self.input_shape).astype(self.dtype)
        return input_data
