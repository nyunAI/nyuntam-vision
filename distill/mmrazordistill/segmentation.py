import os
from mmengine.config import Config
from .utils import *
from .main import MMRazorDistill


class MMRazorDistillSegmentation(MMRazorDistill):
    def __init__(self, teacher_model, model, loaders=None, **kwargs):
        super().__init__(teacher_model, model, loaders, **kwargs)

    def write_config(self):
        if self.method == "pkd_seg":
            config = return_pkd_seg_config(
                self.teacher_config, self.teacher_pth, self.student_config
            )
        else:
            raise ValueError(
                f"Required Distillation Algorithm {self.method} is not supported"
            )
        with open(f"{self.cache_path}/current_distill_config.py", "w") as f:
            f.write(config)
        return

    def customize_config(self):
        self.write_config()
        cfg = Config.fromfile(f"{self.cache_path}/current_distill_config.py")
        cfg["data_root"] = os.path.join(self.data_path, "VOCdevkit", "VOC2012")
        if "ade20k" in self.model_name:
            cfg["dataset_type"] = "PascalVOCDataset"
            cfg["test_dataloader"]["batch_size"] = self.batch_size
            cfg["test_dataloader"]["dataset"]["data_root"] = os.path.join(
                self.data_path, "VOCdevkit", "VOC2012"
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
            cfg["train_dataloader"]["batch_size"] = self.batch_size
            cfg["train_dataloader"]["dataset"]["data_root"] = os.path.join(
                self.data_path, "VOCdevkit", "VOC2012"
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
            cfg["val_dataloader"]["batch_size"] = self.batch_size
            cfg["val_dataloader"]["dataset"]["data_root"] = os.path.join(
                self.data_path, "VOCdevkit", "VOC2012"
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
        cfg["train_dataloader"]["batch_size"] = self.batch_size
        cfg["train_dataloader"]["dataset"]["data_root"] = os.path.join(
            self.data_path, "VOCdevkit", "VOC2012"
        )
        cfg["train_dataloader"]["dataset"]["ann_file"] = os.path.join(
            "ImageSets", "Segmentation", "train.txt"
        )
        cfg["train_dataloader"]["dataset"]["data_root"] = os.path.join(
            self.data_path, "VOCdevkit", "VOC2012"
        )
        cfg["train_dataloader"]["dataset"]["ann_file"] = os.path.join(
            "ImageSets", "Segmentation", "train.txt"
        )
        cfg["train_dataloader"]["dataset"]["data_prefix"]["seg_map_path"] = (
            os.path.join("SegmentationClass")
        )
        cfg["val_dataloader"]["batch_size"] = self.batch_size
        cfg["val_dataloader"]["dataset"]["data_root"] = os.path.join(
            self.data_path, "VOCdevkit", "VOC2012"
        )
        cfg["test_dataloader"]["batch_size"] = self.batch_size
        cfg["test_dataloader"]["dataset"]["data_root"] = os.path.join(
            self.data_path, "VOCdevkit", "VOC2012"
        )
        cfg["test_evaluator"]["batch_size"] = self.batch_size
        cfg["input_shape"] = self.input_shape
        cfg["train_cfg"]["max_epochs"] = self.epochs
        cfg.work_dir = self.cache_path
        cfg.dump(f"{self.cache_path}/current_config_new.py")
        return cfg
