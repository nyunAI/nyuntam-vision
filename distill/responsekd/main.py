import numpy as np
import torch
import torch.nn as nn
from .distill import Distillation
from .losses import KDTransferLoss
import os
import sys
import logging
from datetime import datetime

sys.path.append(os.path.abspath(os.path.join("...", "core")))
from core.finetune import train, validate
from tqdm import tqdm
import wandb
import numpy as np
import os
import time
from nyuntam.algorithm import VisionAlgorithm
from trailmet.utils import AverageMeter, accuracy, save_checkpoint, seed_everything


# Hinton's Knowledge Distillation
seed_everything(43)


class KDTransfer(Distillation, VisionAlgorithm):
    """Class to compress model using distillation via KD transfer.

    Parameters
    ----------
        teacher_model (object): Teacher model you want to use.
        student_model (object): Student model you want to use.
        dataloaders (dict): Dictionary with dataloaders for train, val and test. Keys: 'train', 'val', 'test'.
        paraphraser (object): Paraphrase model
        kwargs (object): YAML safe loaded file with information like distill_args(lambda, temperature, etc).
    """

    def __init__(self, teacher_model, student_model, dataloaders, **kwargs):
        super(KDTransfer, self).__init__(**kwargs)
        self.teacher_model = teacher_model
        self.student_model = student_model
        self.dataloaders = dataloaders
        self.kwargs = kwargs
        self.lambda_ = self.kwargs.get("LAMBDA", 0.5)
        self.temperature = self.kwargs.get("TEMPERATURE", 5)
        ce = self.kwargs.get("CRITERION", "CrossEntropyLoss")
        self.ce_loss = eval(f"torch.nn.{ce}()")
        self.kd_loss = KDTransferLoss(self.temperature, "batchmean")

        self.epochs = kwargs.get("DISTILL_EPOCHS", 200)
        self.lr = kwargs.get("DISTILL_LR", 0.1)

        self.wandb_monitor = self.kwargs.get("wandb", "False")
        self.dataset_name = dataloaders["train"].dataset.__class__.__name__
        self.folder_name = kwargs.get("USER_FOLDER", "abc")
        self.model_path = kwargs.get("MODEL_PATH", "models")
        self.logging_path = kwargs.get("LOGGING_PATH", "logs")
        self.job_id = kwargs.get("JOB_ID", "1")
        self.logger = logging.getLogger(__name__)
        self.save = self.model_path
        self.logger.info(f"Experiment Arguments: {self.kwargs}")

        if self.wandb_monitor:
            wandb.init(project="Kompress Response_KD", name=str(self.job_id))
            wandb.config.update(self.kwargs)
        self.distill_args = {
            "TEMPRATURE": self.temperature,
            "EPOCHS": self.epochs,
            "LR": self.lr,
            "LAMBDA": self.lambda_,
            "WEIGHT_DECAY": kwargs.get("DISTILL_WEIGHT_DECAY"),
            "seed": kwargs.get("DISTILL_SEED"),
        }

    def compress_model(self):
        """Function to transfer knowledge from teacher to student."""
        # include teacher training options
        self.distill(
            self.teacher_model,
            self.student_model,
            self.dataloaders,
            **self.distill_args,
        )

        return "None", __name__

    def distill(self, teacher_model, student_model, dataloaders, **kwargs):
        self.logger.info("=====> TRAINING STUDENT NETWORK <=====")
        train_teacher = kwargs.get("TRAINING", True)
        opt = self.kwargs.get("OPTIMIZER", "Adam")
        lr = kwargs.get("LR", 0.1)
        weight_decay = kwargs.get("WEIGHT_DECAY", 0.0005)

        optimizer = eval(f"torch.optim.{opt}")(
            student_model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )

        milestones = kwargs.get("MILESTONES", [60, 120, 160])
        gamma = kwargs.get("GAMMA", 0.2)
        sched = kwargs.get("SCHEDULER", "MultiStepLR")
        scheduler = eval(f"torch.optim.lr_scheduler.{sched}")(
            optimizer, milestones=milestones, gamma=gamma, verbose=False
        )

        criterion = self.criterion
        best_top1_acc = 0

        if train_teacher == True:
            self.kwargs["VALIDATE"] = True
            self.kwargs["SAVE_MODEL"] = True
            self.kwargs["SAVE_PATH"] = self.save
            self.kwargs["VALIDATION_INTERVAL"] = 1
            self.kwargs["EPOCHS"] = self.epochs
            self.teacher_model, acc1, acc5 = train(
                dataloaders["train"],
                dataloaders["val"],
                self.teacher_model,
                __name__,
                self.kwargs,
            )
        for epoch in range(self.epochs):
            t_loss = self.train_one_epoch(
                teacher_model,
                student_model,
                dataloaders["train"],
                criterion,
                optimizer,
                epoch,
            )

            valid_loss, valid_top1_acc, valid_top5_acc = self.test(
                teacher_model, student_model, dataloaders["val"], criterion, epoch
            )

            # use conditions for different schedulers e.g. ReduceLROnPlateau needs scheduler.step(v_loss)
            scheduler.step()

            if valid_top1_acc > best_top1_acc:
                best_top1_acc = valid_top1_acc
                torch.save(self.student_model, os.path.join(self.save, "mds.pt"))

            if self.wandb_monitor:
                wandb.log({"best_top1_acc": best_top1_acc})

    def train_one_epoch(
        self, teacher_model, student_model, dataloader, loss_fn, optimizer, epoch
    ):
        teacher_model.eval()
        student_model.train()

        batch_time = AverageMeter("Time", ":6.3f")
        data_time = AverageMeter("Data", ":6.3f")
        losses = AverageMeter("Loss", ":.4e")

        end = time.time()

        for i, (images, labels) in enumerate(dataloader):
            data_time.update(time.time() - end)
            images = images.to(self.device, dtype=torch.float)
            labels = labels.to(self.device)

            teacher_preds = teacher_model(images)
            student_preds = student_model(images)
            loss = loss_fn(teacher_preds, student_preds, labels)
            n = images.size(0)
            losses.update(loss.item(), n)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_time.update(time.time() - end)
            end = time.time()
            self.logger.info(
                "Training student network Epoch [%d] (%d / %d Steps) (batch time=%2.5fs) (data time=%2.5fs) (loss=%2.5f)"
                % (
                    epoch,
                    (i + 1),
                    len(dataloader),
                    batch_time.val,
                    data_time.val,
                    losses.val,
                )
            )

            if self.wandb_monitor:
                wandb.log(
                    {
                        "train_loss": losses.val,
                    }
                )

        return losses.avg

    def test(self, teacher_model, student_model, dataloader, loss_fn, epoch):
        batch_time = AverageMeter("Time", ":6.3f")
        losses = AverageMeter("Loss", ":.4e")
        top1 = AverageMeter("Acc@1", ":6.2f")
        top5 = AverageMeter("Acc@5", ":6.2f")

        teacher_model.eval()
        student_model.eval()

        with torch.no_grad():
            end = time.time()

            for i, (images, labels) in enumerate(dataloader):
                images = images.to(self.device, dtype=torch.float)
                labels = labels.to(self.device)

                teacher_preds = teacher_model(images)
                student_preds = student_model(images)
                loss = loss_fn(teacher_preds, student_preds, labels)

                pred1, pred5 = accuracy(student_preds, labels, topk=(1, 5))

                n = images.size(0)
                losses.update(loss.item(), n)
                top1.update(pred1[0], n)
                top5.update(pred5[0], n)

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                self.logger.info(
                    "Validating student network Epoch [%d] (%d / %d Steps) (batch time=%2.5fs) (loss=%2.5f) (top1=%2.5f) (top5=%2.5f)"
                    % (
                        epoch,
                        (i + 1),
                        len(dataloader),
                        batch_time.val,
                        losses.val,
                        top1.val,
                        top5.val,
                    )
                )

                if self.wandb_monitor:
                    wandb.log(
                        {
                            "val_loss": losses.val,
                            "val_top1_acc": top1.val,
                            "val_top5_acc": top5.val,
                        }
                    )

        return losses.avg, top1.avg, top5.avg

    def criterion(self, out_t, out_s, labels):
        ce_loss = self.ce_loss(out_s, labels)
        kd_loss = self.kd_loss(out_t, out_s)
        return (
            self.lambda_ * ce_loss
            + (1 - self.lambda_) * (self.temperature**2) * kd_loss
        )
