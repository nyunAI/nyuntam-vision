import torch
import os


class ModelsFactory(object):
    @staticmethod
    def create_model(
        name,
        num_classes=100,
        pretrained=False,
        version="original",
        platform="torchvision",
        **kwargs
    ):
        """Returns the requested model, ready for training/pruning with the
        specified method.

        Args:
            name: model name 'resnet18','resnet50'
            num_classes: number of classes
        Return:
            model object
        """
        cache_dir = os.path.join(kwargs["CACHE_PATH"], "Torch")
        os.makedirs(os.path.join(cache_dir, "Torch"), exist_ok=True)
        torch.hub.set_dir(cache_dir)
        weight = None
        model = None
        if kwargs["IS_TEACHER"] == True:
            custom_model_path = kwargs.get("CUSTOM_TEACHER_MODEL_PATH")
        else:
            custom_model_path = kwargs.get("CUSTOM_MODEL_PATH", "")
        if (
            custom_model_path != ""
            and os.listdir(custom_model_path) != []
            and "wds.pt" in os.listdir(custom_model_path)
        ):
            from vision.core.utils.modelutils import load_model_or_weights

            model_or_weight, flag = load_model_or_weights(
                os.path.join(custom_model_path, "wds.pt")
            )
            if flag == 1:
                model = model_or_weight
            else:
                weight = model_or_weight
        if model == None:
            if platform == "huggingface":
                from .huggingface import get_hf_model, get_hf_tokenizer

                model = get_hf_model(
                    name, num_classes, kwargs["CACHE_PATH"], task=kwargs["TASK"]
                )
                # triggers saving the tokenizer in the cache path
                if kwargs["TASK"] == "llm":
                    get_hf_tokenizer(
                        name,
                        kwargs["CACHE_PATH"],
                        save_dir=os.path.join(kwargs["CACHE_PATH"], name),
                        task=kwargs["TASK"],
                    )
            elif platform == "timm":
                from .timm import get_timm_model

                model = get_timm_model(name, num_classes, pretrained)
            else:
                if name in [
                    "resnet18",
                    "resnet20",
                    "resnet32",
                    "resnet50",
                    "resnet56",
                    "resnet101",
                    "resnet110",
                ]:
                    if version == "original":
                        assert "insize" in kwargs, "should provide input size"
                        insize = kwargs["insize"]
                        from trailmet.models.resnet import get_resnet_model

                        model = get_resnet_model(
                            name, num_classes, insize=insize, pretrained=pretrained
                        )

                    elif version == "chip":
                        assert "sparsity" in kwargs, "should provide sparsity for chip"
                        from trailmet.models.resnet_chip import (
                            resnet_50 as resnet_50_chip,
                        )

                        model = resnet_50_chip(
                            sparsity=eval(kwargs["sparsity"]), num_classes=num_classes
                        )
                    elif version == "bireal":
                        assert "insize" in kwargs, "should provide input size"
                        insize = kwargs["insize"]
                        assert "num_fp" in kwargs, "should provide num_fp"
                        num_fp = kwargs["num_fp"]
                        from trailmet.models.resnet_bireal import (
                            make_birealnet18,
                            make_birealnet34,
                            make_birealnet50,
                        )

                        if name == "resnet18":
                            model = make_birealnet18(
                                num_classes=num_classes, insize=insize, num_fp=num_fp
                            )
                        elif name == "resnet34":
                            model = make_birealnet34(
                                num_classes=num_classes, insize=insize, num_fp=num_fp
                            )
                        elif name == "resnet50":
                            model = make_birealnet50(
                                num_classes=num_classes, insize=insize, num_fp=num_fp
                            )
                        else:
                            raise Exception(
                                "unknown model {} for BirealNet, available .models_bnnbn are (resnet18, resnet34, resnet50)".format(
                                    name
                                )
                            )

                elif name in ["mobilenetv2"]:
                    if version == "original":
                        from trailmet.models.mobilenet import (
                            get_mobilenet as get_mobilenet_normal,
                        )

                        model = get_mobilenet_normal(name, num_classes, **kwargs)
                    elif version == "bireal":
                        from trailmet.models.mobilenetv2_bireal import (
                            get_mobilenet as get_mobilenet_bireal,
                        )

                        assert "num_fp" in kwargs, "should provide num_fp"
                        num_fp = kwargs["num_fp"]
                        model = get_mobilenet_bireal(num_classes, num_fp=num_fp)
                    else:
                        raise Exception("unknown model {}".format(name))
                elif platform == "mmdet":
                    from .mmdet import get_mmdet_model

                    model = get_mmdet_model(name, kwargs)
                elif platform == "mmseg":
                    from .mmseg import get_mmseg_model

                    model = get_mmseg_model(name, kwargs)
                elif platform == "mmpose":
                    from .mmpose import get_mmpose_model

                    model = get_mmpose_model(name, kwargs)
                elif platform == "mmyolo":
                    from .mmyolo import get_mmyolo_model

                    model = get_mmyolo_model(name, kwargs)
                else:
                    raise Exception(
                        "unknown model {} or Platform ".format(name, platform)
                    )
            if weight != None:
                if "state_dict" in weight.keys():
                    weight = weight["state_dict"]
                    model.load_state_dict(weight)

        return model
