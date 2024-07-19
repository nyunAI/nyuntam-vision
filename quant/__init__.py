def initialize_initialization(algoname):
    if algoname == "FXQuant":
        from .TorchNativeQuantization import FXQuant

        return FXQuant
    elif algoname == "NNCF":
        from .NNCF import NNCF

        return NNCF
    elif algoname == "TensorRT":
        from .TensorRT import TensorRT

        return TensorRT
    elif algoname == "ONNXQuant":
        from .ONNXQuant import ONNXQuant, DummyDataReader

        return ONNXQuant
    elif algoname == "NNCFQAT":
        from .NNCF import NNCFQAT

        return NNCFQAT

    else:
        return None
