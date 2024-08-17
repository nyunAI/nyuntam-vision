import os
import torch
import torch.nn as nn
from collections import OrderedDict


def correct_model_name(model_name):
    """
    Implemented to  overcome the naming discrepencies when using mim download to get configuration and pre-trained weights.
    """
    # lhs:config_name #rhs: mim download nam
    correct_names = {
        "ssd512_pisa_coco": "pisa_ssd512_coco",
        "ssd300_pisa_coco": "pisa_ssd300_coco",
        "retinanet_x101-32x4d_fpn_pisa_1x_coco": "pisa_retinanet_x101_32x4d_fpn_1x_coco",
        "retinanet-r50_fpn_pisa_1x_coco": "pisa_retinanet_r50_fpn_1x_coco",
        "mask-rcnn_r50_fpn_pisa_1x_coco": "pisa_mask_rcnn_r50_fpn_1x_coco",
        "faster-rcnn_x101-32x4d_fpn_pisa_1x_coco": "pisa_faster_rcnn_x101_32x4d_fpn_1x_coco",
        "faster-rcnn_r50_fpn_pisa_1x_coco": "pisa_faster_rcnn_r50_fpn_1x_coco",
        "yolact_r50_1xb8-55e_coco": "yolact_r50_1x8_coco",
        "yolact_r50_8xb8-55e_coco": "yolact_r50_8x8_coco",
        "yolact_r101_1xb8-55e_coco": "yolact_r101_1x8_coco",
        "yolox_tiny_8xb8-300e_coco": "yolox_tiny_8x8_300e_coco",
        "yolox_s_8xb8-300e_coco": "yolox_s_8x8_300e_coco",
        "yolox_l_8xb8-300e_coco": "yolox_l_8x8_300e_coco",
        "yolox_nano_8xb8-300e_coco": "yolox_nano_8x8_300e_coco",
        "yolox_x_8xb8-300e_coco": "yolox_x_8x8_300e_coco",
        "yolof_r50-c5_8xb8-1x_coco": "yolof_r50_c5_8x8_1x_coco",
        "yolov3_d53_8xb8-320-273e_coco": "yolov3_d53_320_273e_coco",
        "yolov3_d53_8xb8-ms-416-273e_coco": "yolov3_d53_mstrain-416_273e_coco",
        "yolov3_d53_8xb8-ms-608-273e_coco": "yolov3_d53_mstrain-608_273e_coco",
    }
    if model_name in correct_names.keys():
        return correct_names[model_name]
    else:
        return model_name


def replace_all_instances(
    config, config_key, new_value, create_additional_parameters={}
):
    from mmengine.config import Config
    from mmengine.config.config import ConfigDict

    """
    Edits all occurances in OpenMM configuration files with the provided key to the specified values
    Parameters:
    config : Loaded mmdet config object (Type: ConfigDict)
    config_key: The key of the config parameter to be updated.
    new_value: The new value to be updated
    additional_parameters: Any new parameters to be added at the same level of the parameter to be replaced.
    """

    def change_value(dic, key, value, additional_parameters={}):
        for k, v in dic.items():
            if k == key and type(dic[k]) == type(value):
                dic[k] = value
                if list(additional_parameters.keys()) != []:
                    for k, v in additional_parameters.items():
                        dic[k] = v
            elif isinstance(v, ConfigDict):
                change_value(v, key, value, additional_parameters)

    for k in config.keys():
        if isinstance(config[k], ConfigDict):
            change_value(config[k], config_key, new_value, create_additional_parameters)
        elif k == config_key:
            config[k] = new_value
    return config


def get_metainfo_coco(data_path):
    """
    OpenMM Libraries requires a modified metainfo to support COCO dataloading for custom datasets.
    """
    from pycocotools.coco import COCO

    ann_file = os.path.join(data_path, "annotations/instances_val2017.json")
    coco = COCO(ann_file)
    categories = coco.loadCats(coco.getCatIds())
    category_names = [category["name"] for category in categories]
    metainfo = dict(classes=category_names)
    return metainfo


def modify_head_classification(model, model_name, num_classes):
    """
    Modifies the head of classification models loaded from timm and huggingface to support the number of classes of the loaded custom datasets
    """
    import torch
    import torch.nn as nn

    layer_names = [name for name, _ in model.named_children()]
    if "head" in layer_names:
        nc = [i for i in model.head.named_children()]
        if nc == []:
            setattr(model, "head", nn.Linear(model.head.in_features, num_classes))
        else:
            if "fc" in model.head.named_children():
                setattr(
                    model.head, "fc", nn.Linear(model.head.fc.in_features, num_classes)
                )
    elif "fc" in layer_names:
        setattr(model, "fc", nn.Linear(model.fc.in_features, num_classes))
    elif "classifier" in layer_names:
        setattr(
            model, "classifier", nn.Linear(model.classifier.in_features, num_classes)
        )
    elif "vanillanet" in model_name:
        model.switch_to_deploy()
        model.cls[2] = nn.Conv2d(
            model.cls[2].in_channels,
            num_classes,
            kernel_size=model.cls[2].kernel_size,
            stride=model.cls[2].stride,
        )
    else:
        raise ValueError(
            f"Not able to find the last fc layer from the layer list {layer_names}"
        )
    return model


def get_state_dict_or_model(loaded):
    """
    Identifies if loaded file is a state dict or a nn.module
    """
    if isinstance(loaded, OrderedDict, dict):
        return "STATE_DICT"
    elif isinstance(loaded, nn.Module):  ## add imports
        return "MODEL"


def init_annfile(cfg, data_path):
    if "ann_file" in [i for i in cfg["test_evaluator"].keys()]:
        cfg["test_evaluator"]["ann_file"] = os.path.join(
            data_path, "annotations/instances_val2017.json"
        )
    else:
        cfg["test_evaluator"]["dataset"]["ann_file"] = os.path.join(
            data_path, "annotations/instances_val2017.json"
        )
    if "ann_file" in [i for i in cfg["val_evaluator"].keys()]:
        cfg["val_evaluator"]["ann_file"] = os.path.join(
            data_path, "annotations/instances_val2017.json"
        )
    else:
        cfg["val_evaluator"]["dataset"]["ann_file"] = os.path.join(
            data_path, "annotations/instances_val2017.json"
        )
    return cfg
