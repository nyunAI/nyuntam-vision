def initialize_initialization(algoname):
    if algoname == "TorchPrune":
        from .torchprune import TorchPrune

        return TorchPrune
    elif algoname == "MMRazorPrune":
        from .mmrazorprune import MMrazorPrune
        return MMrazorPrune