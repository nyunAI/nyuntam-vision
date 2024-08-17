from glob import glob
import os
import shutil
from .main import NNCFQAT
from vision.quant.nncf.ptq.utils import write_deploy_cfg, build_mmdeploy_config
from vision.core.utils.mmutils import create_input_image, customize_config
from .utils import build_quantization_config


class NNCFQATObjectDetection(NNCFQAT):
    def __init__(self, model, loaders=None, **kwargs):
        super().__init__(model, loaders, **kwargs)
        self.iou_threshold = kwargs.get("IOU", 0.65)
        self.score_threshold = kwargs.get("SCORE_THRESHOLD", 0.03)
        self.confidence_threshold = kwargs.get("CONFIDENCE_THRESHOLD", 0.005)
        self.keep_top_k = kwargs.get("MAX_BBOX_PER_IMG", 100)
        self.max_box = kwargs.get("MAX_BBOX_PER_CLS", 100)
        self.pre_top_k = kwargs.get("NMS_PRE", 1000)
        self.model_path = kwargs.get("JOB_PATH", "")
        self.custom_model_path = kwargs.get("CUSTOM_MODEL_PATH", "")
        self.data_path = kwargs.get("DATA_PATH", "")
        self.data_path = os.path.join(self.data_path, "root")
        self.epochs = kwargs.get("EPOCHS", 1)
        self.opt = kwargs.get("OPTIMIZER", "SGD")
        self.lr = kwargs.get("LEARNING_RATE", 0.0001)
        self.momentum = kwargs.get("MOMENTUM", 0.9)
        self.scheduler = kwargs.get("LR_SCHEDULER", "ConstantLR")
        self.factor = kwargs.get("LR_SCHEDULER_FACTOR", 1)
        self.val_interval = kwargs.get("VALIDATION_INTERVAL", 1)
        self.weight_decay = kwargs.get("WEIGHT_DECAY", 0.0005)
        self.work_path = os.getcwd()

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
        runner = customize_config(
            config, self.data_path, self.model_path, self.batch_size, self.cache_path
        )
        quant_config_path = f"{self.cache_path}/current_quant_final.py"
        os.system(
            f"python {self.work_path}vision/core/utils/mmrazortrain.py {quant_config_path} --work-dir {self.job_path}"
        )
        self.quantized_pth_location = None
        if "last_checkpoint" in os.listdir(self.model_path):
            with open(os.path.join(self.model_path, "last_checkpoint"), "r") as f:
                self.quantized_pth_location = f.readline()
                self.logger.info(
                    f"Fake Quantized pth is present at {self.quantized_pth_location}"
                )
                print(f"Fake Quantized pth is present at {self.quantized_pth_location}")

        if self.quantized_pth_location == None:
            raise Exception("Fake Quantization Unsuccessful, checkpoint not found")
        self.logger.info("Fake Quantization Successful")
        # deply config
        build_mmdeploy_config(self.imsize, self.cache_path)
        create_input_image(self.loaders["test"], self.cache_path)
        openvino_config_path = f"{self.cache_path}/current_openvino_deploy_config.py"
        demo_img_path = f"{self.cache_path}/demo_image.png"
        os.system(
            f"python {self.work_path}/vision/core/utils/mmrazordeploy.py {openvino_config_path} {quant_config_path} {self.quantized_pth_location} {demo_img_path}"
        )
        self.logger.info("Deployment Successful")
        shutil.move("end2end.xml", os.path.join(self.model_path, "mds.xml"))
        shutil.move("end2end.bin", os.path.join(self.model_path, "mds.bin"))
        return runner.model, __name__
