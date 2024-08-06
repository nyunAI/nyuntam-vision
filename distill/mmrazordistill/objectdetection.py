import os
from mmengine.config import Config
from .utils import *
from vision.core.utils.mmutils import get_metainfo_coco
from .main import MMRazorDistill


class MMRazorDistillObjectDetection(MMRazorDistill):
    def __init__(self, teacher_model, model, loaders=None, **kwargs):
        super().__init__(teacher_model, model, loaders, **kwargs)

    def write_config(self):
        if self.method == "cwd":
            config = return_cwd_config(
                self.teacher_config, self.teacher_pth, self.student_config
            )
        elif self.method == "pkd":
            config = return_pkd_config(
                self.teacher_config, self.teacher_pth, self.student_config
            )
        elif self.method == "pkd_yolo":
            config = return_pkd_mmyolo_config(
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
        data_path = os.path.join(self.data_path, "root")
        cfg["train_dataloader"]["dataset"]["data_root"] = data_path
        cfg["train_dataloader"]["dataset"]["metainfo"] = get_metainfo_coco(
            os.path.join(self.data_path, "root")
        )
        cfg["val_dataloader"]["dataset"]["data_root"] = data_path
        cfg["val_dataloader"]["dataset"]["metainfo"] = get_metainfo_coco(
            os.path.join(self.data_path, "root")
        )
        cfg["test_dataloader"]["dataset"]["data_root"] = data_path
        cfg["test_dataloader"]["dataset"]["metainfo"] = get_metainfo_coco(
            os.path.join(self.data_path, "root")
        )
        cfg["train_dataloader"]["batch_size"] = self.batch_size
        cfg["val_dataloader"]["batch_size"] = self.batch_size
        cfg["test_dataloader"]["batch_size"] = self.batch_size
        cfg["test_evaluator"]["ann_file"] = os.path.join(
            data_path, "annotations/instances_val2017.json"
        )
        cfg["val_evaluator"]["ann_file"] = os.path.join(
            data_path, "annotations/instances_val2017.json"
        )
        cfg["input_shape"] = self.input_shape
        cfg["train_cfg"]["max_epochs"] = self.epochs
        cfg["train_cfg"]["val_interval"] = self.validation_interval
        cfg.work_dir = self.cache_path
        cfg.dump(f"{self.cache_path}/current_config_new.py")
        return cfg
