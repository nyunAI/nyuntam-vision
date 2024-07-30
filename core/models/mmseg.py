import os

# Create a folder for downloaded checkpoints in the user folder
import subprocess
from glob import glob
from mmdet.utils import setup_cache_size_limit_of_dynamo
from mmengine.config import Config
import shutil
from mmengine.registry import RUNNERS
from mmengine.runner import Runner


def get_mmseg_model(name, kwargs):
    model_name = name
    cache_path = kwargs.get("CACHE_PATH", "")
    launcher = kwargs.get("LAUNCHER", "none")
    intermediate_path = os.path.join(cache_path, f"intermediate_{model_name}")
    amp = kwargs.get("AMP", False)
    auto_scale_lr = kwargs.get("AUTO_SCALE_LR", False)
    qat = kwargs.get("QAT", False)
    batch_size = kwargs.get("BATCH_SIZE", 32)
    epochs = kwargs.get("EPOCHS", 10)
    cache_path = kwargs.get("CACHE_PATH", "")
    num_classes = kwargs.get("NUM_CLASSES", 21)

    if not qat:
        if not os.path.exists(intermediate_path):

            os.makedirs(intermediate_path)
            # Download the weights there
            cpath = os.getcwd()
            os.chdir(intermediate_path)
            os.system(f"mim download mmsegmentation --config {model_name} --dest .")
            os.chdir(cpath)

        setup_cache_size_limit_of_dynamo()
        cfg_path = glob(f"{intermediate_path}/*.py")[0]
        cfg = Config.fromfile(cfg_path)
        cfg.launcher = launcher

        changes = {"data_root": kwargs.get("DATA_PATH", "")}
        cfg.merge_from_dict(changes)
        cfg.work_dir = os.path.join(cache_path)
        if amp is True:
            cfg.optim_wrapper.type = "AmpOptimWrapper"
            cfg.optim_wrapper.loss_scale = "dynamic"

        if auto_scale_lr:
            if (
                "auto_scale_lr" in cfg
                and "enable" in cfg.auto_scale_lr
                and "base_batch_size" in cfg.auto_scale_lr
            ):
                cfg.auto_scale_lr.enable = True
            else:
                raise RuntimeError(
                    'Can not find "auto_scale_lr" or '
                    '"auto_scale_lr.enable" or '
                    '"auto_scale_lr.base_batch_size" in your'
                    " configuration file."
                )
        # Load the runner
        if "runner_type" not in cfg:
            # build the default runner
            runner = Runner.from_cfg(cfg)
        else:
            # build customized runner from the registry
            # if 'runner_type' is set in the cfg
            runner = RUNNERS.build(cfg)
        # Return The runner
        return runner
    else:
        if not os.path.exists(intermediate_path):

            os.makedirs(intermediate_path)
            # Download the weights there
            cpath = os.getcwd()
            os.chdir(intermediate_path)
            os.system(f"mim download mmsegmentation --config {model_name} --dest .")
            os.chdir(cpath)

        setup_cache_size_limit_of_dynamo()
        cfg_path = glob(f"{intermediate_path}/*.py")[0]
        cfg = Config.fromfile(cfg_path)
        data_path = kwargs.get("DATA_PATH", "")
        cfg.dump(os.path.join(cache_path, "modified_cfg.py"))
        cfg["data_root"] = os.path.join(data_path, "VOCdevkit", "VOC2012")
        if "ade20k" in model_name:
            cfg["dataset_type"] = "PascalVOCDataset"
            cfg["test_dataloader"]["batch_size"] = batch_size
            cfg["test_dataloader"]["dataset"]["data_root"] = os.path.join(
                data_path, "VOCdevkit", "VOC2012"
            )
            cfg["test_dataloader"]["dataset"]["type"] = "PascalVOCDataset"
            cfg["test_dataloader"]["dataset"]["pipeline"][2][
                "reduce_zero_label"
            ] = False
            cfg["test_dataloader"]["dataset"]["ann_file"] = os.path.join(
                "ImageSets", "Segmentation", "val.txt"
            )
            cfg["test_dataloader"]["dataset"]["data_prefix"]["img_path"] = "JPEGImages"
            cfg["test_dataloader"]["dataset"]["data_prefix"][
                "seg_map_path"
            ] = "SegmentationClass"
            cfg["test_pipeline"][2]["reduce_zero_label"] = False
            cfg["train_dataloader"]["batch_size"] = batch_size
            cfg["train_dataloader"]["dataset"]["data_root"] = os.path.join(
                data_path, "VOCdevkit", "VOC2012"
            )
            cfg["train_dataloader"]["dataset"]["type"] = "PascalVOCDataset"
            cfg["train_dataloader"]["dataset"]["pipeline"][1][
                "reduce_zero_label"
            ] = False
            cfg["train_dataloader"]["dataset"]["data_prefix"]["img_path"] = "JPEGImages"
            cfg["train_dataloader"]["dataset"]["data_prefix"][
                "seg_map_path"
            ] = "SegmentationClass"
            cfg["train_dataloader"]["dataset"]["ann_file"] = os.path.join(
                "ImageSets", "Segmentation", "val.txt"
            )
            cfg["train_pipeline"][1]["reduce_zero_label"] = False
            cfg["val_dataloader"]["batch_size"] = batch_size
            cfg["val_dataloader"]["dataset"]["data_root"] = os.path.join(
                data_path, "VOCdevkit", "VOC2012"
            )
            cfg["val_dataloader"]["dataset"]["type"] = "PascalVOCDataset"
            cfg["val_dataloader"]["dataset"]["pipeline"][2]["reduce_zero_label"] = False
            cfg["val_dataloader"]["dataset"]["data_prefix"]["img_path"] = "JPEGImages"
            cfg["val_dataloader"]["dataset"]["data_prefix"][
                "seg_map_path"
            ] = "SegmentationClass"
            cfg["val_dataloader"]["dataset"]["ann_file"] = os.path.join(
                "ImageSets", "Segmentation", "val.txt"
            )
            cfg["model"]["decode_head"]["num_classes"] = num_classes
        else:
            cfg["dataset_train"]["data_root"] = os.path.join(
                data_path, "VOCdevkit", "VOC2012"
            )
            cfg["dataset_train"]["ann_file"] = os.path.join(
                data_path,
                "VOCdevkit",
                "VOC2012",
                "ImageSets",
                "Segmentation",
                "train.txt",
            )
            cfg["dataset_aug"]["data_root"] = os.path.join(
                data_path, "VOCdevkit", "VOC2012"
            )
            cfg["dataset_aug"]["ann_file"] = os.path.join(
                data_path,
                "VOCdevkit",
                "VOC2012",
                "ImageSets",
                "Segmentation",
                "train.txt",
            )
            cfg["dataset_aug"]["data_prefix"]["seg_map_path"] = os.path.join(
                "SegmentationClass"
            )
            cfg["train_dataloader"]["batch_size"] = batch_size
            cfg["train_dataloader"]["dataset"]["datasets"][0]["data_root"] = (
                os.path.join(data_path, "VOCdevkit", "VOC2012")
            )
            cfg["train_dataloader"]["dataset"]["datasets"][0]["ann_file"] = (
                os.path.join("ImageSets", "Segmentation", "train.txt")
            )
            cfg["train_dataloader"]["dataset"]["datasets"][1]["data_root"] = (
                os.path.join(data_path, "VOCdevkit", "VOC2012")
            )
            cfg["train_dataloader"]["dataset"]["datasets"][1]["ann_file"] = (
                os.path.join("ImageSets", "Segmentation", "train.txt")
            )
            cfg["train_dataloader"]["dataset"]["datasets"][1]["data_prefix"][
                "seg_map_path"
            ] = os.path.join("SegmentationClass")
            cfg["val_dataloader"]["batch_size"] = batch_size
            cfg["val_dataloader"]["dataset"]["data_root"] = os.path.join(
                data_path, "VOCdevkit", "VOC2012"
            )
            # cfg['val_dataloader']['dataset']['ann_file'] = os.path.join(data_path, 'VOCdevkit','VOC2012','ImageSets','Segmentation','val.txt')
            cfg["test_dataloader"]["batch_size"] = batch_size
            cfg["test_dataloader"]["dataset"]["data_root"] = os.path.join(
                data_path, "VOCdevkit", "VOC2012"
            )
            # cfg['test_dataloader']['dataset']['ann_file'] = os.path.join(data_path, 'VOCdevkit','VOC2012','ImageSets','Segmentation','val.txt')
            cfg["test_evaluator"]["batch_size"] = batch_size
        cfg.launcher = launcher
        cfg.dump(os.path.join(cache_path, "modified_cfg.py"))
        cfg.work_dir = os.path.join(cache_path)

        if amp is True:
            cfg.optim_wrapper.type = "AmpOptimWrapper"
            cfg.optim_wrapper.loss_scale = "dynamic"

        if auto_scale_lr:
            if (
                "auto_scale_lr" in cfg
                and "enable" in cfg.auto_scale_lr
                and "base_batch_size" in cfg.auto_scale_lr
            ):
                cfg.auto_scale_lr.enable = True
            else:
                raise RuntimeError(
                    'Can not find "auto_scale_lr" or '
                    '"auto_scale_lr.enable" or '
                    '"auto_scale_lr.base_batch_size" in your'
                    " configuration file."
                )
        # Load the runner
        if "runner_type" not in cfg:
            # build the default runner
            runner = Runner.from_cfg(cfg)
        else:
            # build customized runner from the registry
            # if 'runner_type' is set in the cfg
            runner = RUNNERS.build(cfg)
        # Return The runner
        return runner
