import os
from glob import glob
from mmdet.utils import setup_cache_size_limit_of_dynamo
from mmengine.config import Config
from mmengine.config.config import ConfigDict
import shutil
from mmengine.registry import RUNNERS
from mmengine.runner import Runner
from ..utils.modelutils import (
    replace_all_instances,
    correct_model_name,
    get_metainfo_coco,
    init_annfile,
)


def get_mmdet_model(name, kwargs):
    model_name = name
    cache_path = kwargs.get("CACHE_PATH", "")
    launcher = kwargs.get("LAUNCHER", "none")
    intermediate_path = os.path.join(cache_path, f"intermediate_{model_name}")
    amp = kwargs.get("AMP", False)
    auto_scale_lr = kwargs.get("AUTO_SCALE_LR", False)
    batch_size = kwargs.get("BATCH_SIZE", 32)
    epochs = kwargs.get("EPOCHS", 10)
    cache_path = kwargs.get("CACHE_PATH", "")
    lr = kwargs.get("LEARNING_RATE", 0.0001)
    default_post_processing = kwargs.get("USE_DEFAULT_POST_PROCESSING", True)
    if not os.path.exists(intermediate_path):
        os.makedirs(intermediate_path)
        # Download the weights there
        cpath = os.getcwd()
        os.chdir(intermediate_path)
        os.system(
            f"mim download mmdet --config {correct_model_name(model_name)} --dest ."
        )
        os.chdir(cpath)
    setup_cache_size_limit_of_dynamo()
    cfg_path = glob(f"{intermediate_path}/*.py")[0]
    cfg = Config.fromfile(cfg_path)
    cfg.launcher = launcher
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
    data_path = kwargs.get("DATA_PATH", "")
    job_path = kwargs.get("JOB_PATH", "")
    data_path = os.path.join(data_path, "root")
    iou = kwargs.get("IOU", 0.65)
    score_thr = kwargs.get("SCORE_THRESHOLD", 0.03)
    max_per_img = kwargs.get("MAX_BBOX_PER_IMG", 100)
    min_bbox_size = kwargs.get("MIN_BBOX_SIZE", 0)
    nms_pre = kwargs.get("NMS_PRE", 1000)
    metainfo = get_metainfo_coco(data_path)
    ann_file = os.path.join("annotations/instances_val2017.json")
    changes = {"data_root": kwargs.get("DATA_PATH", "")}
    cfg.merge_from_dict(changes)
    cfg.work_dir = os.path.join(job_path)
    cfg["optim_wrapper"]["optimizer"]["lr"] = lr

    cfg = replace_all_instances(
        cfg, "data_root", data_path, create_additional_parameters={"metainfo": metainfo}
    )
    cfg = init_annfile(cfg, data_path)
    cfg = replace_all_instances(cfg, "max_epochs", epochs)
    cfg = replace_all_instances(cfg, "batch_size", batch_size)
    cfg = replace_all_instances(cfg, "base_batch_size", batch_size)
    cfg = replace_all_instances(cfg, "num_classes", len(metainfo["classes"]))
    cfg = replace_all_instances(cfg, "num_classes", len(metainfo["classes"]))
    if default_post_processing == False:
        cfg = replace_all_instances(cfg, "nms", dict(type="nms", iou_threshold=iou))
        cfg = replace_all_instances(cfg, "score_thr", score_thr)
        cfg = replace_all_instances(cfg, "min_bbox_size", min_bbox_size)
        cfg = replace_all_instances(cfg, "nms_pre", nms_pre)
        cfg = replace_all_instances(cfg, "max_per_img", max_per_img)

    if kwargs["ALGORITHM"] in ["NNCFQAT", "NNCF", "TensorRTQAT"]:
        if "paramwise_cfg" in cfg["optim_wrapper"]:
            del cfg["optim_wrapper"]["paramwise_cfg"]
    if kwargs["ALGORITHM"] in ["MMRazorPrune"]:
        if "data_root" in cfg["train_dataset"]["dataset"]:
            cfg["train_dataset"]["dataset"][
                "ann_file"
            ] = "annotations/instances_val2017.json"
            # del cfg["train_dataset"]["dataset"]["data_root"]
    cfg.dump(os.path.join(cache_path, "modified_cfg.py"))

    cfg.work_dir = os.path.join(cache_path)
    # Load the runner
    if "runner_type" not in cfg:
        # build the default runner
        runner = Runner.from_cfg(cfg)
    else:
        runner = RUNNERS.build(cfg)
    # Return The runner
    return runner
