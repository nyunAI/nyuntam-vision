import os
import shutil
from glob import glob
from .main import NNCF
from vision.core.utils.mmutils import customize_config, create_input_image
from .utils import write_deploy_cfg, build_quantization_config, build_mmdeploy_config

class NNCFObjectDetection(NNCF):
    def __init__(self, model, loaders=None, **kwargs):
        super().__init__(model, loaders, **kwargs)
        self.platform = kwargs.get("PLATFORM", "mmdet")
        self.custom_model_path = kwargs.get("CUSTOM_MODEL_PATH", "")
        self.cache_path = kwargs.get("CACHE_PATH", "")
        self.data_path = kwargs.get("DATA_PATH", "")
        self.data_path = os.path.join(self.data_path, "root")
        self.batch_size = kwargs.get("BATCH_SIZE", 8)
        self.iou_threshold = kwargs.get("IOU", 0.65)
        self.score_threshold = kwargs.get("SCORE_THRESHOLD", 0.03)
        self.confidence_threshold = kwargs.get("CONFIDENCE_THRESHOLD", 0.005)
        self.keep_top_k = kwargs.get("MAX_BBOX_PER_IMG", 100)
        self.max_box = kwargs.get("MAX_BBOX_PER_CLS", 100)
        self.pre_top_k = kwargs.get("NMS_PRE", 1000)

    def list_subdirectories(self, directory):
        subdirectories = [
            d
            for d in os.listdir(directory)
            if os.path.isdir(os.path.join(directory, d))
        ]
        return subdirectories

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
        )
        config = build_quantization_config(
            self.ckpt_path,
            self.cache_path,
        )
        runner = customize_config(
            config, self.data_path, self.model_path, self.batch_size
        )
        runner.test()
        self.logger.info("Fake Quantization Successful")
        folders = self.list_subdirectories(self.model_path)
        for folder in folders:
            if "model_ptq.pth" in os.listdir(os.path.join(self.model_path, folder)):
                self.quantized_pth_location = os.path.join(
                    self.model_path, folder, "model_ptq.pth"
                )
                break
        # deply config
        build_mmdeploy_config(self.imsize)
        create_input_image(self.loaders["test"])
        os.system(
            f"python vision/core/utils/mmrazordeploy.py current_openvino_deploy_config.py current_quant_config.py {self.quantized_pth_location} demo_image.png"
        )
        self.logger.info("Deployment Successful")
        shutil.move("/workspace/nyuntam/end2end.xml", os.path.join(self.model_path, "mds.xml"))
        shutil.move("/workspace/nyuntam/end2end.bin", os.path.join(self.model_path, "mds.bin"))
        return runner.model, __name__
