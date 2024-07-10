import torch
import torchvision
import torch_pruning as tp
import numpy as np
import os
import wandb
from datetime import datetime
import logging
import sys

sys.path.append(os.path.abspath(os.path.join("..", "logging_kompress")))
from logging_kompress import define_logger
from ultralytics.nn.modules import Detect
from ultralytics.yolo.utils import yaml_load
from ultralytics.yolo.utils.checks import check_yaml
from ultralytics.yolo.utils.torch_utils import initialize_weights
from ultralytics import YOLO
import time
import timm
from transformers.models.vit.modeling_vit import ViTSelfAttention, ViTSelfOutput
from .torchprune_utils import replace_c2f_with_c2f_v2, train_v2, forward_timm_vit
from copy import deepcopy
from trailmet.utils import AverageMeter, accuracy

sys.path.append(os.path.abspath(os.path.join("..", "finetune")))
logger = logging.getLogger(__name__)
from finetune import train


class TorchPrune:
    def __init__(self, model, loaders, **kwargs):
        self.kwargs = kwargs
        self.wandb = kwargs.get("wandb", True)
        self.task = kwargs.get("TASK", "image_classification")
        self.dataset = kwargs.get("DATASET_NAME", "CIFAR10")
        self.device = kwargs.get("DEVICE", "cuda:0")
        self.model_name = kwargs.get("MODEL", "resnet50")
        self.model = model
        self.loaders = loaders
        self.imp_name = kwargs.get("GROUP_IMPORTANCE", "GroupTaylorImportance")
        self.pruner_name = kwargs.get("PRUNER_NAME", "MetaPruner")
        self.ch_sparsity = kwargs.get("SPARSITY", 0.5)
        self.classes = kwargs.get("NUM_CLASSES", 10)
        self.loss_fn_name = kwargs.get("LOSS_FN", "CrossEntropyLoss")
        self.optimizer_name = kwargs.get("OPTIMIZER_NAME", "Adam")
        self.epochs = kwargs.get("EPOCHS", 10)
        self.is_iterative = kwargs.get("ITERATIVE", False)
        self.root = kwargs.get("DATA_PATH", "data/")
        self.max_drop = kwargs.get("MAX_DROP", 0.2)
        self.target_prune_rate = kwargs.get("TARGET_PRUNE_RATE", 0.5)
        self.iterative_steps = kwargs.get("ITERATIVE_STEPS", 16)
        self.batch_size = kwargs.get("BATCH_SIZE", 32)
        self.platform = kwargs.get("PLATFORM", "torchvision")
        self.bottleneck = kwargs.get("BOTTLENECK", False)
        self.num_heads = {}
        self.pruning_ratio = self.ch_sparsity
        self.prune_num_heads = kwargs.get("PRUNE_NUM_HEADS", False)
        self.head_prune_ratio = kwargs.get("HEAD_PRUNE_RATIO", 0.5)
        self.imsize = kwargs.get("imsize", 32)
        self.to_train = kwargs.get("TRAINING", True)
        self.obj_transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize(size=(self.imsize, self.imsize)),
                torchvision.transforms.ToTensor(),
            ]
        )
        self.example_inputs, self.example_outputs = next(iter(loaders["test"]))
        self.folder_name = kwargs.get("USER_FOLDER", "abc")
        self.model_path = kwargs.get("MODEL_PATH", "models")
        self.logging_path = kwargs.get("LOGGING_PATH", "logs")

        self.logger = define_logger(
            __name__, self.logging_path
        )
        if "yolov8" in self.model_name:
            from torchvision import transforms
            import math
            import ultralytics

            ultralytics.LOGGER = logging.getLogger(__name__)
            self.imsize = kwargs.get("insize", "32")
            self.is_iterative = True

            self.ch_sparsity = 1 - math.pow(
                (1 - self.target_prune_rate), 1 / self.iterative_steps
            )

        try:
            eval(f"tp.importance.{self.imp_name}")
        except:
            raise Exception(f"No Importance {self.imp_name}")

        self.imp = eval(f"tp.importance.{self.imp_name}()")

        try:
            eval(f"tp.pruner.{self.pruner_name}")
        except:
            raise Exception(f"No Pruner Named {self.pruner_name}")
        try:
            eval(f"torch.nn.{self.loss_fn_name}")
        except:
            raise Exception(f"No Loss Function {self.loss_fn_name} found in torch.nn")

        self.loss_fn = eval(f"torch.nn.{self.loss_fn_name}")()

        try:
            eval(f"torch.optim.{self.optimizer_name}")
        except:
            raise Exception(f"No Optimizer {self.optimizer_name} found in torch.optim")
        if self.task == "image_classification":
            self.optimizer = eval(f"torch.optim.{self.optimizer_name}")(
                self.model.parameters()
            )

        self.logger.info(f"Experiment Arguments: {self.kwargs}")
        self.job_id = kwargs.get("JOB_ID","1")
        if self.wandb:
            wandb.init(project="Kompress TorchPrune", name=str(self.job_id))
            wandb.config.update(self.kwargs)
    def init_pruner(self):
        self.init_ignore_layers()
        if "yolov8" in self.model_name:
            self.pruner = eval(f"tp.pruner.{self.pruner_name}")(
                self.model.model,
                self.example_inputs,
                importance=self.imp,
                ch_sparsity=self.ch_sparsity,
                ignored_layers=self.ignored_layers,
            )
        elif "vit" in self.model_name:
            self.pruner = eval(f"tp.pruner.{self.pruner_name}")(
                self.model,
                self.example_inputs,
                global_pruning=False,
                importance=self.imp,
                ch_sparsity=self.pruning_ratio,
                ignored_layers=self.ignored_layers,
                num_heads=self.num_heads,
                prune_num_heads=self.prune_num_heads,
                prune_head_dims=self.prune_num_heads,
                head_pruning_ratio=self.head_prune_ratio,
            )

        else:
            self.pruner = eval(f"tp.pruner.{self.pruner_name}")(
                self.model,
                self.example_inputs,
                importance=self.imp,
                ch_sparsity=self.ch_sparsity,
                ignored_layers=self.ignored_layers,
            )

    def init_ignore_layers(self):
        ignored_layers = []
        if self.task == "image_classification":
            for m in self.model.modules():
                if isinstance(m, torch.nn.Linear) and m.out_features == self.classes:
                    ignored_layers.append(m)

        if "yolov8" in self.model_name:
            for m in self.model.modules():
                if isinstance(m, (Detect,)):
                    ignored_layers.append(m)

        if "ssd" in self.model_name:
            ignored_layers.append(self.model.head)

        if self.model_name == "raft_large":
            ignored_layers.extend(
                [
                    self.model.corr_block,
                    self.model.update_block,
                    self.model.mask_predictor,
                ]
            )
        if "fasterrcnn" in self.model_name:
            ignored_layers.extend(
                [
                    self.model.rpn.head.cls_logits,
                    self.model.rpn.head.bbox_pred,
                    self.model.backbone.fpn,
                    self.model.roi_heads,
                ]
            )

        ## Removable Ignored Layers not implemented in our model section, but can retain for future integrations
        if self.model_name == "fcos_resnet50_fpn":
            ignored_layers.extend(
                [
                    self.model.head.classification_head.cls_logits,
                    self.model.head.regression_head.bbox_reg,
                    self.model.head.regression_head.bbox_ctrness,
                ]
            )
        if self.model_name == "keypointrcnn_resnet50_fpn":
            ignored_layers.extend(
                [
                    self.model.rpn.head.cls_logits,
                    self.model.backbone.fpn.layer_blocks,
                    self.model.rpn.head.bbox_pred,
                    self.model.roi_heads.box_head,
                    self.model.roi_heads.box_predictor,
                    self.model.roi_heads.keypoint_predictor,
                ]
            )
        if self.model_name == "maskrcnn_resnet50_fpn_v2":
            ignored_layers.extend(
                [
                    self.model.rpn.head.cls_logits,
                    self.model.rpn.head.bbox_pred,
                    self.model.roi_heads.box_predictor,
                    self.model.roi_heads.mask_predictor,
                ]
            )
        if self.model_name == "retinanet_resnet50_fpn_v2":
            ignored_layers.extend(
                [
                    self.model.head.classification_head.cls_logits,
                    self.model.head.regression_head.bbox_reg,
                ]
            )
        if "vit" in self.model_name and self.platform == "timm":
            ignored_layers.append(self.model.head)
            for m in self.model.modules():
                if isinstance(m, timm.models.vision_transformer.Attention):
                    m.forward = forward_timm_vit.__get__(
                        m, timm.models.vision_transformer.Attention
                    )
                    self.num_heads[m.qkv] = m.num_heads
                if self.bottleneck and isinstance(
                    m, timm.models.vision_transformer.Mlp
                ):
                    m.forward = forward_timm_vit.__get__(
                        m, timm.models.vision_transformer.Attention
                    )
                    self.num_heads[m.qkv] = m.num_heads
                if self.bottleneck and isinstance(
                    m, timm.models.vision_transformer.Mlp
                ):
                    ignored_layers.append(m.fc2)

        if "vit" in self.model_name and self.platform == "huggingface":
            ignored_layers = [self.model.classifier]
            # All heads should be pruned simultaneously, so we group channels by head.
            for m in self.model.modules():
                if isinstance(m, ViTSelfAttention):
                    self.num_heads[m.query] = m.num_attention_heads
                    self.num_heads[m.key] = m.num_attention_heads
                    self.num_heads[m.value] = m.num_attention_heads
                if self.bottleneck and isinstance(m, ViTSelfOutput):
                    ignored_layers.append(m.dense)
        self.ignored_layers = ignored_layers

    def set_model_params(self):
        self.model.__setattr__("train_v2", train_v2.__get__(self.model))
        self.model.model.train()
        replace_c2f_with_c2f_v2(self.model.model)
        initialize_weights(self.model.model)

    def train_sparse(self):
        self.logger.info("Beginning Sparse Training")
        batch_time = AverageMeter("Time", ":6.3f")
        losses = AverageMeter("Loss", ":.4e")
        top1 = AverageMeter("Acc@1", ":6.2f")
        top5 = AverageMeter("Acc@5", ":6.2f")
        for epoch in range(self.epochs):
            self.model.to(self.device)
            self.model.train()
            for i, (data, target) in enumerate(self.loaders["train"]):
                start = time.time()
                data, target = data.to(self.device), target.to(self.device)
                self.optimizer.zero_grad()
                out = self.model(data)
                if type(out) != torch.Tensor:
                    if "logits" in dir(out):
                        out = out.logits
                op = torch.nn.functional.one_hot(target, num_classes=self.classes)
                loss = self.loss_fn(out, op.float())
                loss.backward()
                self.pruner.regularize(self.model)
                self.optimizer.step()
                pred1, pred5 = accuracy(out, target, topk=(1, 5))
                n = data.size(0)
                losses.update(loss.item(), n)
                top1.update(pred1[0], n)
                top5.update(pred5[0], n)
                end = time.time()
                batch_time.update(end - start, n)
                # self.logger.info(f"Epoch [{epoch}/{self.epochs}] loss = {loss}")
                self.logger.info(
                    "Epoch [%d] (%d / %d Steps) (batch time=%2.5fs) (loss=%2.5f) (top1=%2.5f) (top5=%2.5f)"
                    % (
                        epoch,
                        (i + 1),
                        len(self.loaders["train"]),
                        batch_time.val,
                        losses.val,
                        top1.val,
                        top5.val,
                    )
                )

                if self.wandb:
                    wandb.log({"loss": loss})
            self.logger.info("Sparse Training Completed")

    def perform_pruning(self):
        self.init_pruner()
        base_macs, base_nparams = tp.utils.count_ops_and_params(
            self.model, self.example_inputs
        )
        self.logger.info(
            f"Operations Before Pruning = {base_macs}  \n Parameters Before Pruning= {base_nparams}"
        )

        if isinstance(
            self.pruner, (tp.pruner.BNScalePruner, tp.pruner.GroupNormPruner)
        ):
            self.train_sparse()
        if isinstance(self.imp, tp.importance.GroupTaylorImportance):
            outputs = self.model(self.example_inputs)
            op = torch.nn.functional.one_hot(
                self.example_outputs, num_classes=self.classes
            )
            loss = self.loss_fn(op.float().to(self.device), outputs.float().to(self.device))
            loss.backward()

        self.pruner.step()
        macs, nparams = tp.utils.count_ops_and_params(
            self.model.to(self.device), self.example_inputs.to(self.device)
        )
        self.logger.info(
            f"Operations After Pruning = {macs}  \n Parameters After Pruning= {nparams}"
        )
        return self.model

    def init_yolov8(self):
        self.model.__setattr__("train_v2", train_v2.__get__(self.model))
        self.model.model.train()
        replace_c2f_with_c2f_v2(self.model.model)
        initialize_weights(self.model.model)

    def iterative_pruning_yolo(self):
        if "yolov8" in self.model_name:
            self.init_yolov8()
        pruning_cfg = yaml_load(
            check_yaml(
                os.path.join(
                    os.getcwd(), "algorithms_kompress/prune", "yolo_train_prune.yaml"
                )
            )
        )
        batch_size = self.batch_size

        pruning_cfg["data"] = os.path.join(
            self.root, "supporting_yaml_coco_format.yaml"
        )
        pruning_cfg["epochs"] = self.epochs

        for name, param in self.model.model.named_parameters():
            param.requires_grad = True

        example_inputs = torch.randn(1, 3, self.imsize, self.imsize).to(self.device)
        macs_list, nparams_list, map_list, pruned_map_list = [], [], [], []
        base_macs, base_nparams = tp.utils.count_ops_and_params(
            self.model.model.to(self.device), example_inputs
        )

        pruning_cfg["name"] = f"baseline_val"
        pruning_cfg["batch"] = 1
        validation_model = deepcopy(self.model)
        metric = validation_model.val(**pruning_cfg)
        init_map = metric.box.map

        self.logger.info(
            f"Before Pruning: MACs={base_macs / 1e9: .5f} G, #Params={base_nparams / 1e6: .5f} M, mAP={init_map: .5f}"
        )
        self.init_pruner()
        for i in range(self.iterative_steps):
            self.model.model.train()
            example_inputs = example_inputs.to(self.device)
            for name, param in self.model.model.named_parameters():
                param.requires_grad = True
            self.pruner.step()

            pruning_cfg["name"] = f"step_{i}_pre_val"
            pruning_cfg["batch"] = self.batch_size
            validation_model.model = deepcopy(self.model.model)
            metric = validation_model.val(**pruning_cfg)
            pruned_map = metric.box.map
            pruned_macs, pruned_nparams = tp.utils.count_ops_and_params(
                self.pruner.model.to(self.device), example_inputs.to(self.device)
            )
            current_speed_up = float(macs_list[0]) / pruned_macs
            self.logger.info(
                f"After pruning iter {i + 1}: MACs={pruned_macs / 1e9} G, #Params={pruned_nparams / 1e6} M, "
                f"mAP={pruned_map}, speed up={current_speed_up}"
            )

            for name, param in self.model.model.named_parameters():
                param.requires_grad = True
            pruning_cfg["name"] = f"step_{i}_finetune"
            pruning_cfg["batch"] = batch_size
            self.model.train_v2(pruning=True, **pruning_cfg)

            pruning_cfg["name"] = f"step_{i}_post_val"
            pruning_cfg["batch"] = 1
            validation_model = YOLO(self.model.trainer.best)
            metric = validation_model.val(**pruning_cfg)
            current_map = metric.box.map
            self.logger.info(f"After fine tuning mAP={current_map}")

            macs_list.append(pruned_macs)
            nparams_list.append(pruned_nparams / base_nparams * 100)
            pruned_map_list.append(pruned_map)
            map_list.append(current_map)

            del self.pruner
            self.init_pruner()
            if init_map - current_map > self.max_drop:
                self.logger.info("Pruning early stop")
                break

        return validation_model

    def save_model(self, pruned_model):
        torch.save(pruned_model, f"{self.model_path}/mds.pt")

    def compress_model(self):
        if self.is_iterative == False:
            model = self.perform_pruning()
            if self.to_train:
                model = train(
                    self.loaders["train"],
                    self.loaders["val"],
                    model,
                    __name__,
                    self.kwargs,
                )
            self.save_model(model)
        else:
            if "yolov8" in self.model_name:
                model = self.iterative_pruning_yolo()

        return model, __name__
