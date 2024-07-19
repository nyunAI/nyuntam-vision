from transformers import AutoModelForImageClassification, AutoModelForCausalLM
import os
from HeadModification import modify_head_classification


def get_hf_model(model_name, num_classes, cache_dir, task="image_classification"):
    if task == "image_classification":
        os.makedirs(os.path.join(cache_dir, "HuggingFace"), exist_ok=True)
        model = AutoModelForImageClassification.from_pretrained(
            model_name, cache_dir=os.path.join(cache_dir, "HuggingFace")
        )

        model = modify_head_classification(model, model_name, num_classes)
    elif task == "llm":
        model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=cache_dir)
    else:
        raise NotImplementedError

    return model


def get_hf_tokenizer(model_name, cache_dir, save_dir: str = None, task="llm"):
    if task == "llm":
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            model_name, cache_dir=cache_dir, use_fast=True
        )
    else:
        raise NotImplementedError

    if save_dir is not None:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
        tokenizer.save_pretrained(save_dir)
    return tokenizer
