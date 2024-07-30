from .main import NNCF
import nncf
import openvino as ov
import os
import uuid
import shutil
import torch
from nyuntam.vision.core.finetune import train


class NNCFClassifcation(NNCF):
    def __init__(self, model, loaders=None, **kwargs):

        super().__init__(model, loaders, **kwargs)
        self.transformer = kwargs.get("TRANSFORMER", False)
        self.to_train = kwargs.get("TRAINING", True)
        if self.transformer:
            self.model_type = nncf.parameters.ModelType.TRANSFORMER
        else:
            self.model_type = None

    def transform_fn_image_only(self, data_item):
        images, _ = data_item
        return images

    def compress_model(self):

        if self.to_train:
            self.model, _, _ = train(
                self.loaders["train"],
                self.loaders["test"],
                self.model,
                __name__,
                self.kwargs,
            )
        input_fp32 = torch.randn(1, 3, self.imsize, self.imsize)
        calibration_dataset = nncf.Dataset(
            self.loaders["test"], self.transform_fn_image_only
        )
        quantized_model = nncf.quantize(
            self.model, calibration_dataset, model_type=self.model_type
        )
        os.makedirs(f"{self.model_path}/onnx_temp_models", exist_ok=True)
        onnx_model_path = f"{self.model_path}/onnx_temp_models/{str(uuid.uuid4())}"
        torch.onnx.export(
            quantized_model.to("cpu"), input_fp32.to("cpu"), onnx_model_path
        )
        ov_quantized_model = ov.convert_model(onnx_model_path)
        ov.serialize(
            ov_quantized_model,
            f"{self.model_path}/mds.xml",
        )
        shutil.rmtree(f"{self.model_path}/onnx_temp_models")
        return ov_quantized_model, __name__
