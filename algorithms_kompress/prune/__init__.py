#from .torchprune import TorchPrune
#from .mmrazor import MMRazor


def initialize_initialization(algoname):
    if algoname == "TorchPrune":
        from .torchprune import TorchPrune
        return TorchPrune