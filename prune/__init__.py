
def initialize_initialization(algoname):
    if algoname == "TorchPrune":
        from .TorchPrune import TorchPrune
        return TorchPrune