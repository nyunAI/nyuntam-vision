def initialize_initialization(algoname, task):
    if algoname == "FXQuant":
        from .torch import FXQuant

        return FXQuant
    elif algoname == "NNCF":
        if task == "image_classification":
            from .nncf import NNCFClassifcation
            return NNCFClassifcation
        elif task == "object_detection":
            from .nncf import NNCFObjectDetection
            return NNCFObjectDetection

    elif algoname == "TensorRT":
        from .tensorrt import TensorRT

        return TensorRT
    elif algoname == "ONNXQuant":
        from .onnx import ONNXQuant, DummyDataReader

        return ONNXQuant
    elif algoname == "NNCFQAT":
        from .nncf import NNCFQAT

        return NNCFQAT

    else:
        return None
