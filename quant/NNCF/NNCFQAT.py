import nncf.torch
import nncf
import wandb
import os
import sys
from nncf import NNCFConfig
from nncf.torch import create_compressed_model, register_default_init_args

sys.path.append(os.path.abspath(os.path.join("...", "core")))
from core.finetune import train


class NNCFQAT:
    def __init__(self, model, loaders=None, **kwargs):
        self.kwargs = kwargs
        self.model = model
        self.loaders = loaders
        self.wandb = kwargs.get("wandb", True)
        self.dataset_name = kwargs.get("DATASET_NAME", "CIFAR10")
        self.model_name = kwargs.get("MODEL", "resnet50")
        self.imsize = kwargs.get("insize", "32")
        self.transformer = kwargs.get("TRANSFORMER", False)
        self.to_train = kwargs.get("TRAINING", True)
        self.batch_size = kwargs.get("BATCH_SIZE", 8)
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
        self.qat = kwargs.get("QAT", False)
        self.platform = kwargs.get("PLATFORM", "mmdet")

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
