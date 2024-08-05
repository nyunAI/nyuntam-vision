import nncf.torch
import nncf
from nncf import NNCFConfig
from nncf.torch import create_compressed_model, register_default_init_args
from vision.core.finetune import train
from .main import NNCFQAT


class NNCFQATClassification(NNCFQAT):
    def __init__(self, model, loaders=None, **kwargs):
        super().__init__(model, loaders, **kwargs)
        self.transformer = kwargs.get("TRANSFORMER", False)
        self.to_train = kwargs.get("TRAINING", True)
        if self.transformer:
            self.model_type = nncf.parameters.ModelType.TRANSFORMER
        else:
            self.model_type = None

    def compress_model(self):
        if self.to_train:
            self.model, _, _ = train(
                self.loaders["train"],
                self.loaders["test"],
                self.model,
                __name__,
                self.kwargs,
            )

        # Apply QAT using NNCF
        nncf_config_dict = {
            "input_info": {
                "sample_size": [self.batch_size, 3, self.imsize, self.imsize]
            },  # Update with your input size
            "compression": {
                "algorithm": "quantization",  # 8-bit quantization with default settings
            },
        }
        nncf_config = NNCFConfig.from_dict(nncf_config_dict)
        nncf_config = register_default_init_args(nncf_config, self.loaders["train"])

        compression_ctrl, model = create_compressed_model(self.model, nncf_config)

        # Fine-tune the model (optional)
        self.kwargs["QAT_STEP"] = True
        self.kwargs["QAT_SCHEDULER"] = compression_ctrl
        self.model, _, _ = train(
            self.loaders["train"],
            self.loaders["test"],
            self.model,
            __name__,
            self.kwargs,
        )
        # Export quantized model
        compression_ctrl.export_model(f"{self.model_path}/mds.onnx")
        return model, __name__
