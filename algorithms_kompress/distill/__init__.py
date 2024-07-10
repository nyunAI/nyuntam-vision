#from .attention_transfer import AttentionTransfer
# from .distill import Distillation
# #from .factor_transfer import FactorTransfer
# from .losses import KDTransferLoss, RkdDistance, RKdAngle
# from .response_kd import KDTransfer
# from .mmrazor_distill import MMRazorDistill
# from .mmrazor_configs import *
#from .rkd import RKDTransfer

def initialize_initialization(algoname):
    if algoname == "KDTransfer":
        from .distill import Distillation
        from .losses import KDTransferLoss, RkdDistance, RKdAngle
        from .response_kd import KDTransfer
        return KDTransfer
