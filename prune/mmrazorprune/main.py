import subprocess
import wandb
import os
import subprocess
from mmengine.config import Config
from collections import OrderedDict
from glob import glob
import logging
import torch
from mmengine.runner import Runner
from nyuntam.algorithm import VisionAlgorithm


class MMrazorPrune(VisionAlgorithm):
    def __init__(self, model, loaders=None, **kwargs):
        self.kwargs = kwargs
        self.model = model
        self.loaders = loaders
        self.wandb = kwargs.get("wandb", True)
        self.task = kwargs.get("TASK", "image_classification")
        self.device = kwargs.get("DEVICE", "cuda:0")
        self.dataset_name = kwargs.get("DATASET_NAME", "CIFAR10")
        self.model_name = kwargs.get("MODEL", "resnet50")
        self.imsize = kwargs.get("insize", "32")
        self.batch_size = kwargs.get("BATCH_SIZE", 32)
        self.to_train = kwargs.get("TRAINING", True)
        self.folder_name = kwargs.get("USER_FOLDER", "abc")
        self.model_path = kwargs.get("MODEL_PATH", "models")
        self.logging_path = kwargs.get("LOGGING_PATH", "logs")
        self.data_path = kwargs.get("DATA_PATH", "data")
        self.cache_path = kwargs.get("CACHE_PATH", ".cache")
        self.logger = logging.getLogger(__name__)
        self.job_id = kwargs.get("JOB_ID", "1")
        self.platform = kwargs.get("PLATFORM", "torchvision")
        self.interval = kwargs.get("INTERVAL", 10)
        self.norm_type = kwargs.get("NORM_TYPE", "act")
        self.lr_ratio = kwargs.get("LR_RATIO", 0.1)
        self.target_flop_ratio = kwargs.get("TARGET_FLOP_RATIO", 0.5)
        self.insize = kwargs.get("insize", 32)
        self.input_shape = (1, 3, self.insize, self.insize)
        self.epochs = kwargs.get("EPOCHS", 1)
        self.prune_epochs = kwargs.get("PRUNE_EPOCHS", 50)
        self.job_path = kwargs.get("JOB_PATH")
        self.finetune_lr = kwargs.get("FINETUNE_LR_PRUNE", 0.0001)
        cache_path = kwargs.get("CACHE_PATH", "")
        intermediate_path = os.path.join(cache_path, f"intermediate_{self.model_name}")
        self.logger.info(f"Experiment Arguments: {self.kwargs}")
        if not os.path.exists(intermediate_path):

            os.makedirs(intermediate_path)
            # Download the weights there
            cpath = os.getcwd()
            os.chdir(intermediate_path)
            if self.platform == "mmdet":
                os.system(f"mim download mmdet --config {self.model_name} --dest .")
            elif self.platform == "mmyolo":
                os.system(f"mim download mmyolo --config {self.model_name} --dest .")
            elif self.platform == "mmpose":
                os.system(f"mim download mmpose --config {self.model_name} --dest .")
            os.chdir(cpath)
        self.cfg_path = glob(f"{intermediate_path}/*.py")[0]
        self.pth_path = glob(f"{intermediate_path}/*.pth")[0]
        if self.wandb:
            wandb.init(project="Kompress MMRazor", name=str(self.job_id))
            wandb.config.update(self.kwargs)

    def write_config(self):
        config = f"""_base_ = 'modified_cfg.py'
pretrained_path = '{self.pth_path}'  # noqa

interval = {self.interval}
normalization_type = '{self.norm_type}'
lr_ratio = {self.lr_ratio}
data_root = 'data'
target_flop_ratio = {self.target_flop_ratio}
input_shape = (1, 3, {self.insize}, {self.insize})
##############################################################################

architecture = _base_.model

if hasattr(_base_, 'data_preprocessor'):
    architecture.update({{'data_preprocessor': _base_.data_preprocessor}})
    data_preprocessor = {{}}

architecture.init_cfg = dict(type='Pretrained', checkpoint=pretrained_path)
architecture['_scope_'] = _base_.default_scope
architecture.backbone.frozen_stages = -1

model = dict(
    _delete_=True,
    _scope_='mmrazor',
    type='GroupFisherAlgorithm',
    architecture=architecture,
    interval=interval,
    mutator=dict(
        type='GroupFisherChannelMutator',
        parse_cfg=dict(type='ChannelAnalyzer', tracer_type='FxTracer'),
        channel_unit_cfg=dict(
            type='GroupFisherChannelUnit',
            default_args=dict(normalization_type=normalization_type, ),
        ),
    ),
)

model_wrapper_cfg = dict(
    type='mmrazor.GroupFisherDDP',
    broadcast_buffers=False,
)

optim_wrapper = dict(
    optimizer=dict(lr=_base_.optim_wrapper.optimizer.lr * lr_ratio))

custom_hooks = getattr(_base_, 'custom_hooks', []) + [
    dict(type='mmrazor.PruningStructureHook'),
    dict(
        type='mmrazor.ResourceInfoHook',
        interval=interval,
        demo_input=dict(
            type='mmrazor.DefaultDemoInput',
            input_shape=input_shape,
        ),
        save_ckpt_thr=[target_flop_ratio],
    ),
]

        """
        with open(f"{self.cache_path}/current_fisher_config.py", "w") as f:
            f.write(config)
        return

    def write_finetune_cfg(self, flop_path):
        config = f'''#############################################################################
"""# You have to fill these args.

_base_(str): The path to your pruning config file.
pruned_path (str): The path to the checkpoint of the pruned model.
finetune_lr (float): The lr rate to finetune. Usually, we directly use the lr
    rate of the pretrain.
"""

_base_ = 'current_config_new.py'
pruned_path = '{flop_path}'
finetune_lr = {self.finetune_lr}
##############################################################################

algorithm = _base_.model
algorithm.init_cfg = dict(type='Pretrained', checkpoint=pruned_path)

model = dict(
    _delete_=True,
    _scope_='mmrazor',
    type='GroupFisherSubModel',
    algorithm=algorithm,
)

# restore lr
optim_wrapper = dict(optimizer=dict(lr=finetune_lr))

# remove pruning related hooks
custom_hooks = _base_.custom_hooks[:-2]

# delete ddp
model_wrapper_cfg = None'''
        with open(f"{self.cache_path}/current_fisher_finetune_config.py", "w") as f:
            f.write(config)
        return

    def customize_config(self):
        self.write_config()
        cfg = Config.fromfile(f"{self.cache_path}/current_fisher_config.py")
        cfg["train_cfg"]["max_epochs"] = self.prune_epochs
        cfg.work_dir = self.job_path
        cfg.dump("current_config_new.py")

        return cfg

    def customize_finetune_config(self):
        flops_file_path = None
        files = glob(f"{self.job_path}/*")
        for file in files:
            if "flops" in file.split("/")[-1]:
                flops_file_path = file
        if flops_file_path == None:
            raise RuntimeError(
                f"Cannot Find Model Post Pruned Stage flops_XX.pth at {self.job_path}"
            )
        self.write_finetune_cfg(flops_file_path)
        cfg = Config.fromfile(f"{self.cache_path}/current_fisher_finetune_config.py")
        cfg["train_cfg"]["max_epochs"] = self.epochs
        cfg.dump(f"{self.cache_path}/current_config_finetune_new.py")
        return cfg

    def save_model(self, pruned_model):
        with open(f"{self.job_path}/last_checkpoint", "f") as f:
            file_path = f.readline()
        sd = torch.load(file_path)
        sd = sd["state_dict"]
        new_sd = OrderedDict()
        for k in sd.keys():
            if "architecture" in k:
                new_k = ".".join(k.split(".")[1:])
                new_sd[new_k] = sd[k]
            else:
                new_sd[new_k] = sd[k]
        torch_store = {"state_dict": new_sd}
        torch.save(torch_store, f"{self.job_path}/mds.pt")
        torch.save(pruned_model, f"{self.job_path}/mds_full.pt")

    def run_command_realtime(self, command):
        try:
            process = subprocess.Popen(
                command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
            )
            for line in process.stdout:
                self.logger.info(line.rstrip())
            process.communicate()

        except subprocess.CalledProcessError as e:
            raise Exception(e)

    def compress_model(self):
        cfg = self.customize_config()
        command = [
            "python",
            "vision/core/utils/mmrazortrain.py",
            f"{self.cache_path}current_config_new.py",
            "--work-dir",
            self.job_path,
        ]
        self.run_command_realtime(command)
        self.logger.info("Pruning Completed")
        self.logger.info(f"Begining Finetuning for {self.epochs} Epochs")
        finetune_cfg = self.customize_finetune_config()
        runner_finetune = Runner.from_cfg(finetune_cfg)
        runner_finetune.train()
        self.logger.info(f"Finetuning Completed")
        self.save_model(runner_finetune.model)
        self.logger.info(f"Saved Pruned Model at {self.job_path}/mds.pt")
        return "None", __name__
