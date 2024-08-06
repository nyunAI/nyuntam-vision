import os
from .cocopose import CocoPoseEstimationDataset

# from .vocdetection import VOCDetectionDataset


class DatasetFactory(object):
    """This class forms the generic wrapper for the different dataset classes.

    The module includes utilities to load datasets, including methods to load
    and fetch popular reference datasets.
    """

    @staticmethod
    def create_dataset(**kwargs):
        """
        Args:
            name(string): dataset name
            root(string):  Root directory of dataset where directory

        Return:
            dataset(tuple): dataset
        """
        assert "name" in kwargs, "should provide dataset name"
        name = kwargs["name"]
        assert "root" in kwargs, "should provide dataset root"
        if "COCOPOSEESTIMATION" == name:
            obj_dfactory = CocoPoseEstimationDataset(**kwargs)
        # elif "VOCDETECTION" == name:
        #     obj_dfactory = VOCDetectionDataset(**kwargs)
        elif "CUSTOMCOCOPOSEESTIMATION" == name:
            obj_dfactory = CocoPoseEstimationDataset(
                download=False, custom=True, **kwargs
            )
        # elif "CUSTOMVOCDETECTION" == name:
        #     obj_dfactory = VOCDetectionDataset(download=False, **kwargs)
        else:
            raise Exception(f"unknown dataset{kwargs['name']}")

        dataset = obj_dfactory.stack_dataset()
        dataset = obj_dfactory.build_dict_info()

        return dataset
