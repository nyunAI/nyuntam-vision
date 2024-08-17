import torch
from torch.ao.quantization import (
    get_default_qconfig_mapping,
    QConfigMapping,
)
import torch.ao.quantization.quantize_fx as quantize_fx
import copy
import wandb
import logging
from tqdm import tqdm
from vision.core.finetune import train
from nyuntam.algorithm import VisionAlgorithm


class FXQuant(VisionAlgorithm):
    def __init__(self, model, loaders=None, **kwargs):
        self.model = model
        self.loaders = loaders
        self.device = kwargs.get("DEVICE", "cuda:0")
        self.wandb = kwargs.get("wandb", True)
        self.dataset_name = kwargs.get("DATASET_NAME", "CIFAR10")
        self.choice = kwargs.get("choice", "static")
        self.to_train = kwargs.get("TRAINING", True)
        self.folder_name = kwargs.get("USER_FOLDER", "abc")
        self.model_path = kwargs.get("MODEL_PATH", "models")
        self.logging_path = kwargs.get("LOGGING_PATH", "logs")
        self.logger = logging.getLogger(__name__)
        self.quantized_model_path = f"{self.model_path}/mds.pt"

        self.kwargs = kwargs
        self.logger.info(f"Experiment Arguments: {self.kwargs}")
        self.job_id = kwargs.get("JOB_ID", "1")
        if self.wandb:
            wandb.init(project="Kompress FXQuant", name=str(self.job_id))
            wandb.config.update(self.kwargs)

    def weight_only_quantization(
        self,
    ):
        model_to_quantize = copy.deepcopy(self.model)
        model_to_quantize.eval()
        qconfig_mapping = QConfigMapping().set_global(
            torch.ao.quantization.default_dynamic_qconfig
        )
        x, y = next(iter(self.loaders["test"]))
        example_inputs = x
        model_prepared = quantize_fx.prepare_fx(
            model_to_quantize, qconfig_mapping, example_inputs
        )
        self.logger.info("Model Prepared For Quantization")

        model_quantized = quantize_fx.convert_fx(model_prepared)
        self.logger.info("Model Quantized")
        return model_quantized

    def static_quantization(self):
        model_to_quantize = copy.deepcopy(self.model)
        qconfig_mapping = get_default_qconfig_mapping("qnnpack")
        model_to_quantize.eval()
        x, y = next(iter(self.loaders["test"]))
        example_inputs = x
        model_prepared = quantize_fx.prepare_fx(
            model_to_quantize, qconfig_mapping, example_inputs
        )
        self.logger.info("Model Prepared For Quantization")

        self.calibrate(model_prepared, self.loaders["test"])
        model_quantized = quantize_fx.convert_fx(model_prepared)

        self.logger.info("Model Quantized")
        return model_quantized

    def calibrate(self, model, data_loader):
        model.eval()
        self.logger.info("Calibrating")
        with torch.no_grad():
            for image, target in tqdm(data_loader, desc="Calibrating ..."):
                model(image)

        self.logger.info("Calibrated")

    def train_one_epoch(self, model, training_loader, optimizer, loss_fn, epoch_index):
        running_loss = 0.0
        last_loss = 0.0

        for i, data in enumerate(training_loader):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = model(inputs.to(self.device))
            loss = loss_fn(outputs, labels.to(self.device))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 10:
                last_loss = running_loss / 1000  # loss per batch
                self.logger.info(
                    "epoch {} batch {} loss: {}".format(epoch_index, i + 1, last_loss)
                )
                if self.wandb:
                    wandb.log({"loss": last_loss})
                running_loss = 0.0
        return last_loss

    def train(self, model, data_loader, optimizer="adam", loss="ce", epochs=50):
        if optimizer == "adam":
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        else:
            optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
        if loss == "ce":
            loss = torch.nn.CrossEntropyLoss()

        model.train()
        self.logger.info("Training")
        for i in range(epochs):
            ll = self.train_one_epoch(model, data_loader["train"], optimizer, loss, i)

    def quant_aware_training(self):
        model_to_quantize = copy.deepcopy(self.model)
        qconfig_mapping = get_default_qconfig_mapping("qnnpack")
        model_to_quantize.eval()
        x, y = next(iter(self.loaders["test"]))
        example_inputs = x
        model_prepared = quantize_fx.prepare_fx(
            model_to_quantize, qconfig_mapping, example_inputs
        )
        self.logger.info("Model Prepared For Quantization")
        model_prepared.to(self.device)
        self.train(
            model_prepared,
            self.loaders,
            optimizer=self.kwargs.get("OPTIMIZER", "adam"),
            loss=self.kwargs.get("LOSS_FN", "ce"),
            epochs=self.kwargs.get("EPOCHS", 50),
        )
        model_prepared.eval()
        model_quantized = quantize_fx.convert_fx(model_prepared.to("cpu"))
        self.logger.info("Model Quantized")
        return model_quantized

    def fusion_quantization(self):
        model_to_quantize = copy.deepcopy(self.model)
        model_to_quantize.eval()
        model_fused = quantize_fx.fuse_fx(model_to_quantize)

        self.logger.info("Model Quantized")
        return model_fused

    def compress_model(self):
        x, y = next(iter(self.loaders["train"]))
        if self.to_train:
            self.model, _, _ = train(
                self.loaders["train"],
                self.loaders["test"],
                self.model,
                __name__,
                self.kwargs,
            )
            self.model = self.model.to("cpu")
        if self.choice == "weight":
            qm = self.weight_only_quantization()
        elif self.choice == "static":
            qm = self.static_quantization()
        elif self.choice == "fusion":
            qm = self.fusion_quantization()
        elif self.choice == "qat":
            qm = self.quant_aware_training()
        else:
            raise Exception("Wrong Choice Valid Choices = weight,satic,fusion")
            return

        torch.save(qm, self.quantized_model_path)
        return qm, __name__
