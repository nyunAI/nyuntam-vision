import nncf
import openvino as ov
import wandb
import os
import uuid
import sys
import shutil

sys.path.append(os.path.abspath(os.path.join("...", "core")))
import torch
from core.finetune import train
import logging


class NNCF:
    def __init__(self, model, loaders=None, **kwargs):
        self.kwargs = kwargs
        self.model = model
        self.loaders = loaders
        self.wandb = kwargs.get("wandb", True)
        self.imsize = kwargs.get("insize", "32")
        self.transformer = kwargs.get("TRANSFORMER", False)
        self.to_train = kwargs.get("TRAINING", True)
        self.model_path = kwargs.get("MODEL_PATH", "")
        if self.transformer:
            self.model_type = nncf.parameters.ModelType.TRANSFORMER
        else:
            self.model_type = None
        self.model_path = kwargs.get("MODEL_PATH", "models")
        self.logging_path = kwargs.get("LOGGING_PATH", "logs")
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Experiment Arguments: {self.kwargs}")
        self.job_id = kwargs.get("JOB_ID", "1")
        if self.wandb:
            wandb.init(project="Kompress NNCF", name=str(self.job_id))
            wandb.config.update(self.kwargs)

    def transform_fn_image_only(self, data_item):
        images, _ = data_item
        return images

    def compress_classification(self):
        input_fp32 = torch.randn(1, 3, self.imsize, self.imsize)
        calibration_dataset = nncf.Dataset(
            self.loaders["test"], self.transform_fn_image_only
        )
        quantized_model = nncf.quantize(
            self.model, calibration_dataset, model_type=self.model_type
        )
        os.makedirs(f"{self.model_path}/onnx_temp_models", exist_ok=True)
        onnx_model_path = f"{self.model_path}/onnx_temp_models/{str(uuid.uuid4())}"
        onnx_model_path_ori = f"{self.model_path}/onnx_temp_models/{str(uuid.uuid4())}"
        torch.onnx.export(
            quantized_model.to("cpu"), input_fp32.to("cpu"), onnx_model_path
        )
        torch.onnx.export(
            self.model.to("cpu"), input_fp32.to("cpu"), onnx_model_path_ori
        )
        ov_quantized_model = ov.convert_model(onnx_model_path)
        ov.serialize(
            ov_quantized_model,
            f"{self.model_path}/mds.xml",
        )
        shutil.rmtree(f"{self.model_path}/onnx_temp_models")
        return ov_quantized_model

    def compress_model(self):

        if self.to_train:
            self.model, _, _ = train(
                self.loaders["train"],
                self.loaders["test"],
                self.model,
                __name__,
                self.kwargs,
            )
        model = self.compress_classification()
        return model, __name__
