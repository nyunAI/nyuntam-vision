def modify_head_classification(model, model_name,num_classes):
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
        model.cls[2] = nn.Conv2d(model.cls[2].in_channels,num_classes,kernel_size =model.cls[2].kernel_size, stride = model.cls[2].stride )
    else:
        raise ValueError(
            f"Not able to find the last fc layer from the layer list {layer_names}"
        )
    return model