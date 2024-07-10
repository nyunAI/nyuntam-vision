from models_kompress import ModelsFactory
import os

def create_model(model_name, path, **kw):
    model = ModelsFactory.create_model(
        model_name,
        kw.get("NUM_CLASSES", "2"),
        kw.get("PRETRAINED", True),
        version=kw.get("VERSION", "original"),
        platform= kw.get("PLATFORM","torchvision"),
        **kw
    )
    if kw.get("TASK") == "llm":
        model.save_pretrained(path)
    return model
