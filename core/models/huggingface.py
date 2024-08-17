from transformers import AutoModelForImageClassification, AutoModelForCausalLM
import os
from vision.core.utils.modelutils import modify_head_classification


def get_hf_model(model_name, num_classes, cache_dir, task="image_classification"):

    os.makedirs(os.path.join(cache_dir, "HuggingFace"), exist_ok=True)
    model = AutoModelForImageClassification.from_pretrained(
        model_name, cache_dir=os.path.join(cache_dir, "HuggingFace")
    )
    model = modify_head_classification(model, model_name, num_classes)
    return model
