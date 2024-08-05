def initialize_initialization(algoname):
    if algoname == "KDTransfer":
        from .responsekd import Distillation
        from .losses import KDTransferLoss, RkdDistance, RKdAngle
        from .response_kd import KDTransfer

        return KDTransfer
