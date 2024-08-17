import os
from glob import glob
from .utils import (
    make_deploy_config,
    make_deploy_config_mmyolo,
    write_modified_test_loader_config,
)
from vision.core.utils.mmutils import create_input_image
from .main import TensorRT


class TensorRTObjectDetection(TensorRT):
    def __init__(self, model, loaders=None, **kwargs):
        super().__init__(model, loaders, **kwargs)
        self.job_id = kwargs.get("JOB_ID", "1")
        self.platform = kwargs.get("PLATFORM", "torchvision")
        self.cache_path = kwargs.get("CACHE_PATH", "")
        self.intermediate_path = os.path.join(
            self.cache_path, f"intermediate_{self.model_name}"
        )
        self.custom_model_path = kwargs.get("CUSTOM_MODEL_PATH", "")
        self.model = self.model.model

    def convert_tensorrt(self):
        if (
            self.custom_model_path
            and os.listdir(self.custom_model_path) != []
            and "wds.pt" in os.listdir(self.custom_model_path)
        ):
            self.ckpt_path = glob(f"{self.custom_model_path}/*.pth")[0]
        else:
            intermediate_path = os.path.join(
                self.cache_path, f"intermediate_{self.model_name}"
            )
            self.ckpt_path = glob(f"{intermediate_path}/*.pth")[0]
        if self.platform == "mmdet":
            make_deploy_config(self.imsize, self.cache_path)
        elif self.platform == "mmyolo":
            make_deploy_config_mmyolo(self.imsize, self.cache_path)
            write_modified_test_loader_config(
                os.path.join(self.cache_path, "modified_cfg.py"), self.cache_path
            )
        create_input_image(self.loaders["test"], self.cache_path)
        if self.platform == "mmdet":
            os.system(
                f"python vision/core/utils/mmrazordeploy.py {os.path.join(self.cache_path,'current_tensorrt_quant_config.py')} {os.path.join(self.cache_path,'modified_cfg.py')} {self.ckpt_path} {os.path.join(self.cache_path,'demo_img.png')} --device cuda --quant --work-dir {self.model_path}"
            )
        elif self.platform == "mmyolo":
            os.system(
                f"python vision/core/utils/mmrazordeploy.py {os.path.join(self.cache_path,'current_tensorrt_quant_config.py')} {os.path.join(self.cache_path,'modified_pretrain_cfg_tensorrt_quantization.py')} {self.ckpt_path} {os.path.join(self.cache_path,'demo_img.png')} --device cuda --quant --work-dir {self.model_path}"
            )
        self.logger.info("TRT Conversion Success")
        return None

    def compress_model(self):

        self.convert_tensorrt()
        return "None", __name__
