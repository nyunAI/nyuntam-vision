import torch
from datetime import datetime
import copy
import wandb
import logging
import os

logger = logging.getLogger(__name__)


class EMQuant:
    def __init__(self, model, loaders=None, **kwargs):
        self.choice = kwargs.get("choice", "static")
        self.kwargs = kwargs
        self.model = model
        self.loaders = loaders
        self.wandb = kwargs.get("wandb", True)
        self.dataset_name = kwargs.get("DATASET_NAME", "CIFAR10")
        self.log_dir = kwargs.get("log_dir", os.getcwd())
        self.name = "_".join(
            [
                self.dataset_name,
                "EMQUANT",
                datetime.now().strftime("%b-%d_%H-%M-%S"),
            ]
        )

        os.makedirs(f"{self.log_dir}", exist_ok=True)
        self.logger_file = f"{self.log_dir}/{self.name}.log"
        logging.basicConfig(
            filename=self.logger_file,
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H-%M-%S",
            level=logging.INFO,
        )
        logger.info(f"Experiment Arguments: {self.kwargs}")
        if self.wandb:
            wandb.init(project="Trailmet EMQuant", name=self.name)
            wandb.config.update(self.kwargs)

    def static_em(self):
        model_fp32 = copy.deepcopy(self.model)
        model_fp32.eval()
        model_fp32.qconfig = torch.ao.quantization.get_default_qconfig("x86")
        model_fp32_prepared = torch.ao.quantization.prepare(model_fp32)
        logger.info("Model Prepared")
        input_fp32, _ = next(iter(self.loaders["test"]))
        model_fp32_prepared(input_fp32)
        model_int8 = torch.ao.quantization.convert(model_fp32_prepared)
        logger.info("Model Converted")
        return model_int8

    def compress_model(self):
        if self.choice == "static":
            qm = self.static_em()
        else:
            raise Exception("Wrong Choice Valid Choices = satic")

        return qm