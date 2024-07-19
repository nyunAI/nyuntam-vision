def initialize_initialization(algoname):
    if algoname == "KDTransfer":
        from .ResponseKD import Distillation
        from .losses import KDTransferLoss, RkdDistance, RKdAngle
        from .response_kd import KDTransfer

        return KDTransfer


def initialize_initialization(algoname):
    if algoname == "KDTransfer":
        from .ResponseKD import (
            Distillation,
            KDTransferLoss,
            RkdDistance,
            RKdAngle,
            KDTransfer,
        )

        return KDTransfer
