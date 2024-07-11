from typing import Any

def __getattr__(name: str) -> Any:
    # classification
    if name == "ClassificationDatasetFactory":
        from .classification import DatasetFactory as ClassificationDatasetFactory
        return ClassificationDatasetFactory
    else:
        raise AttributeError(f"module {__name__} has no attribute {name}")
