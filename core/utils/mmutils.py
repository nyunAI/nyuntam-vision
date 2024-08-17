from torchvision.utils import save_image
from .modelutils import get_metainfo_coco, replace_all_instances, init_annfile
from mmengine.config import Config
from mmengine.runner import Runner
import os


def create_input_image(loader, cache_path):
    x, y = next(iter(loader))
    if x.shape[0] != 1:
        x = x[0]
    save_image(x, f"{cache_path}/demo_image.png")


def customize_config(config, data_path, model_path, batch_size, cache_path):
    with open(f"{cache_path}/current_quant_config.py", "w") as f:
        f.write(config)
    cfg = Config.fromfile(f"{cache_path}/current_quant_config.py")
    metainfo = get_metainfo_coco(data_path)
    cfg = replace_all_instances(
        cfg, "data_root", data_path, create_additional_parameters={"metainfo": metainfo}
    )
    cfg = init_annfile(cfg, data_path)
    cfg = replace_all_instances(cfg, "batch_size", batch_size)
    cfg.work_dir = os.path.join(model_path)
    runner = Runner.from_cfg(cfg)
    cfg.dump(f"{cache_path}/current_quant_final.py")
    return runner
