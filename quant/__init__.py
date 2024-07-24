def initialize_initialization(algoname):
    if algoname == "FXQuant":
        from .torch import FXQuant

        return FXQuant
    elif algoname == "NNCF":
        from .nncf import NNCF

        return NNCF
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
