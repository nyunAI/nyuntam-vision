from glob import glob
import wandb
import os
import logging
import shutil
from .utils import *
from vision.core.utils.mmutils import create_input_image, customize_config


class TensorRTQAT:
    def __init__(self, model, loaders=None, **kwargs):
        self.kwargs = kwargs
        self.model = model
        self.loaders = loaders
        self.wandb = kwargs.get("wandb", True)
        self.task = kwargs.get("TASK", "image_classification")
        self.batch_size = kwargs.get("BATCH_SIZE", "256")
        self.device = kwargs.get("DEVICE", "cuda:0")
        self.dataset_name = kwargs.get("DATASET_NAME", "CIFAR10")
        self.model_name = kwargs.get("MODEL", "resnet50")
        self.imsize = kwargs.get("insize", "32")
        self.transformer = kwargs.get("TRANSFORMER", False)
        self.to_train = kwargs.get("TRAINING", True)
        self.folder_name = kwargs.get("USER_FOLDER", "abc")
        self.iou_threshold = kwargs.get("IOU", 0.65)
        self.score_threshold = kwargs.get("SCORE_THRESHOLD", 0.03)
        self.confidence_threshold = kwargs.get("CONFIDENCE_THRESHOLD", 0.005)
        self.keep_top_k = kwargs.get("MAX_BBOX_PER_IMG", 100)
        self.max_box = kwargs.get("MAX_BBOX_PER_CLS", 100)
        self.pre_top_k = kwargs.get("NMS_PRE", 1000)
        self.custom_model_path = kwargs.get("CUSTOM_MODEL_PATH", "")
        self.cache_path = kwargs.get("CACHE_PATH", "")
        self.data_path = kwargs.get("DATA_PATH", "")
        self.data_path = os.path.join(self.data_path, "root")
        self.batch_size = kwargs.get("BATCH_SIZE", 8)
        self.model_path = kwargs.get("MODEL_PATH", "")
        self.job_path = kwargs.get("JOB_PATH", "")
        self.epochs = kwargs.get("EPOCHS", 1)
        self.opt = kwargs.get("OPTIMIZER", "SGD")
        self.lr = kwargs.get("LEARNING_RATE", 0.0001)
        self.momentum = kwargs.get("MOMENTUM", 0.9)
        self.scheduler = kwargs.get("LR_SCHEDULER", "ConstantLR")
        self.factor = kwargs.get("LR_SCHEDULER_FACTOR", 1)
        self.val_interval = kwargs.get("VALIDATION_INTERVAL", 1)
        self.weight_decay = kwargs.get("WEIGHT_DECAY", 0.0005)
        self.model_path = kwargs.get("MODEL_PATH", "models")
        self.work_dir = os.getcwd()
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Experiment Arguments: {self.kwargs}")
        self.job_id = kwargs.get("JOB_ID", "1")
        self.fake_quantize_step = kwargs.get("FAKE_QUANTIZED_STEP", True)
        if self.wandb:
            wandb.init(project="Kompress Tensorrt QAT", name=str(self.job_id))
            wandb.config.update(self.kwargs)
        self.qat = kwargs.get("QAT", False)
        self.platform = kwargs.get("PLATFORM", "mmdet")

    def compress_model(self):
        if (
            self.custom_model_path
            and os.listdir(self.custom_model_path) != []
            and "wds.pt" in os.listdir(self.custom_model_path)
        ):
            self.ckpt_path = os.path.join(self.custom_model_path, "wds.pt")
        else:
            intermediate_path = os.path.join(
                self.cache_path, f"intermediate_{self.model_name}"
            )
            self.ckpt_path = glob(f"{intermediate_path}/*.pth")[0]

        write_deploy_cfg(
            self.imsize,
            self.score_threshold,
            self.confidence_threshold,
            self.iou_threshold,
            self.max_box,
            self.pre_top_k,
            self.keep_top_k,
            self.cache_path,
        )
        config = build_quantization_config(
            self.ckpt_path,
            self.cache_path,
            self.epochs,
            self.val_interval,
            self.opt,
            self.momentum,
            self.lr,
            self.scheduler,
            self.factor,
            self.weight_decay,
        )
        customize_config(
            config, self.data_path, self.model_path, self.batch_size, self.cache_path
        )
        quant_config_path = f"{self.cache_path}/current_quant_final.py"

        if self.fake_quantize_step == True:
            os.system(
                f"python {self.work_dir}/vision/core/utils/mmrazortrain.py {quant_config_path} --work-dir {self.job_path}"
            )
            self.quantized_pth_location = None

            if "last_checkpoint" in os.listdir(self.model_path):
                with open(os.path.join(self.model_path, "last_checkpoint"), "r") as f:
                    self.quantized_pth_location = f.readline()
                    self.logger.info(
                        f"Fake Quantized pth is present at {self.quantized_pth_location}"
                    )

            if self.quantized_pth_location == None:
                self.logger.info("Fake Quantization Unsuccessful, checkpoint not found")
                raise Exception("Fake Quantization Unsuccessful, checkpoint not found")
            else:
                self.logger.info("Fake Quantization Successful")
        else:
            # deply config
            self.quantized_pth_location = self.kwargs.get("FAKE_QUANTIZED_PATH", "")

            if self.quantized_pth_location == None:
                self.logger.info(f"Fake Quantized Path is None")
                raise Exception("Fake Quantized Path is None")
            elif not os.path.exists(self.quantized_pth_location):
                self.logger.info(
                    f"Fake Quantized Path is Not Present at {self.quantized_pth_location}"
                )
                raise Exception(
                    f"Fake Quantized Path is Not Present at {self.quantized_pth_location}"
                )
            build_mmdeploy_config(self.imsize, self.cache_path)
            create_input_image(self.loaders["test"], self.cache_path)
            deploy_config_path = f"{self.cache_path}/current_tensorrt_deploy_config.py"
            demo_image_path = f"{self.cache_path}/demo_image.png"
            os.system(
                f"python {self.work_dir}/vision/core/utils/mmrazordeploy.py {deploy_config_path} {quant_config_path} {self.quantized_pth_location} {demo_image_path}"
            )
            self.logger.info("Deployment Successful")
            shutil.move("end2end.xml", os.path.join(self.model_path, "mds.xml"))
            shutil.move("end2end.bin", os.path.join(self.model_path, "mds.bin"))

        return self.model, __name__
