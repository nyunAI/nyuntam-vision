
# nyuntam
from algorithm import Algorithm
from factory import Factory as BaseFactory, FactoryTypes

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from data import prepare_data
from model import create_model
from finetune import validate
import copy
import torch
import importlib
import os
from trailmet.utils.benchmark import ModelBenchmark



class CompressionFactory(BaseFactory):
    """
    Factory to productionize all algorithms defined in trailmet.
    Algorithm specific compression pipeline, arguments and setup can be defined in this class.
    """

    _type: FactoryTypes = FactoryTypes.VISION

    def collate_fn_obj(self, batch):
        images, targets = zip(*batch)
        images = torch.stack(images, 0)
        # targets = torch.stack(targets,0)

        return tuple([images, targets])
    
    def get_algorithm(self, name: str) -> Algorithm:
        algo_type = self.kwargs.get("ALGO_TYPE", "prune")
        module = importlib.import_module(f"algorithms_kompress.{algo_type}")
        loaded_algorithm = getattr(module, "initialize_initialization")(name)
        return loaded_algorithm

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        algo_type = self.kwargs.get("ALGO_TYPE", "prune")
        algorithm = self.kwargs.get("ALGORITHM", "ChipNet")
        loaded_algorithm = self.get_algorithm(algorithm)
        kw = {}
        for k in kwargs.keys():
            if type(kwargs[k]) != type({}):
                kw.update({k: kwargs[k]})
        kw.update(kwargs[algo_type][algorithm])
        self.kw = kw
        self.kw['IS_TEACHER'] = False
        task = self.kw.get("TASK", "image_classification")
        model_name = self.kw.get("MODEL", "resnet50")

        dataset_dict = prepare_data(
            self.kw.get("DATASET_NAME", "cifar10"),
            self.kw.get("DATA_URL"),
            self.kw.get("DATA_PATH"),
            **self.kw
        )

        model = None
        if algo_type == "distill":
            st_name = self.kw.get("MODEL", "resnet18")
            teach_name = self.kw.get("TEACHER_MODEL","")
            student_kw = copy.deepcopy(self.kw)
            student_kw['IS_TEACHER'] = False
            student_model = create_model(
                st_name, self.kw.get("STUDENT_MODEL_PATH", ""), **student_kw
            )
            if self.kw.get("requires_cuda_transfer", False):
                student_model = student_model.cuda()
            teacher_kw = copy.deepcopy(self.kw)
            teacher_kw['IS_TEACHER'] = True
            model = create_model(teach_name, self.kw.get(
                "TEACHER_MODEL", ""), **teacher_kw)

        elif algorithm not in []:
            if os.path.exists(model_name):
                # model is to be loaded from model_name => cache_path := model_name
                self.kw["CACHE_PATH"] = model_name
            else:
                cache_path = os.path.join(kw["CACHE_PATH"], model_name)
                model = create_model(model_name, cache_path, **self.kw)

        self.model = model

        if self.kw.get("requires_cuda_transfer", False):
            model = model.cuda()

        dataloader_dict = {}
        if dataset_dict != None:
            for split in dataset_dict:
                shuffle = True if split == "train" else False
                dataloader_dict[split] = DataLoader(
                        dataset_dict[split],
                        batch_size=self.kw.get("BATCH_SIZE"),
                        shuffle=shuffle,
                        num_workers=self.kw.get("WORKERS", 0),
                        pin_memory=self.kw.get("PIN_MEM", False),
                    )
        self.dataloader_dict = dataloader_dict
        if algo_type == "distill":
            self.algorithm = loaded_algorithm(
                model, student_model, dataloader_dict, **(self.kw)
            )
        elif algorithm in []:
            self.algorithm = loaded_algorithm(**(self.kw))
        else:
            self.algorithm = loaded_algorithm(
                model, dataloader_dict, **(self.kw))
        self.algorithm.log_name = self.kw.get("log_name")

    def __call__(self):
        model2, self.name = self.algorithm.compress_model()
        if self.kwargs['BENCHMARK'] == True:
            self.benchmark_classification(model2)
        return

    def benchmark_classification(self, model2):
        criterion = nn.CrossEntropyLoss()

        model_bench = ModelBenchmark(
            self.model.to("cpu"),
            self.kwargs.get("BATCH_SIZE"),
            self.kwargs.get("insize"),
            device_name="cpu",
        )
        model_bench.benchmark()

        top1_avg_acc, top5_avg_acc = validate(
            self.dataloader_dict["test"], self.model, criterion, self.kwargs
        )
        model_bench = ModelBenchmark(
            model2,  # .to("cpu"),
            self.kwargs.get("BATCH_SIZE"),
            self.kwargs.get("insize"),
            device_name="cpu",
        )
        model_bench.benchmark()

        top1_avg_acc, top5_avg_acc = validate(
            self.dataloader_dict["test"], model2, criterion, self.kwargs
        )
        return self.name
