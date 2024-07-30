import torchvision
from torchvision.datasets.utils import download_and_extract_archive
from .dataset import BaseDataset
import os


class VOCSegmentationDataset(BaseDataset):
    """
    VOC Segmentation

    Expected Directory Structure
    ----------------------------
    Dataset Not found / is of incorrrect fomat at {self.root}. Proper Format :
                                /VOCdevkit
                                    --- /VOC2012
                                    ------ /Annotations
                                    ------ /JPEGImages
                                    ------ / ImageSets
                                    ------ /SegmentationClass
                                    ------ /SegmentationObject

    Parameters
    ----------
        name (string): dataset name 'ImageNet', default=None.
        root (string): Root directory of dataset or will be saved to if download
             is set to True, default=None.
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``. Default=None.
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it, default=None.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again, default=True.
        split_types (list): the possible values of this parameter are "train", "test" and "val".
            If the split_type contains "val", then suffle has to be True, default value is None.
        val_fraction (float): If float, should be between 0.0 and 1.0 and represent
        the proportion of the dataset to include in the val split.
        shuffle (bool): Whether or not to shuffle the data before splitting into val from train,
            default is True. If shuffle is true, there should be 'val' in split_types.
        random_seed (int): RandomState instance, default=None.
    """

    def __init__(
        self,
        name=None,
        root=None,
        transform=None,
        target_transform=None,
        download=True,
        split_types=None,
        val_fraction=0.2,
        shuffle=True,
        random_seed=None,
    ):
        if download == False:
            root = "VOC_Segmentation_Global"
            os.makedirs("VOC_Segmentation_Global", exist_ok=True)
        super(VOCSegmentationDataset, self).__init__(
            name=name,
            root=root,
            transform=transform,
            target_transform=target_transform,
            download=download,
            split_types=split_types,
            val_fraction=val_fraction,
            shuffle=shuffle,
            random_seed=random_seed,
        )

        ## To do: check val_frac is float, else raise error
        ## To do: if shuffle is true, there should be 'val' in train test split
        dataset = torchvision.datasets.VOCSegmentation
        if self.download == False:
            if not all(
                [
                    os.path.exists(
                        os.path.join(self.root, "VOCdevkit", "VOC2012", "JPEGImages")
                    ),
                    os.path.exists(
                        os.path.join(self.root, "VOCdevkit", "VOC2012", "Annotations")
                    ),
                    os.path.exists(
                        os.path.join(self.root, "VOCdevkit", "VOC2012", "ImageSets")
                    ),
                    os.path.exists(
                        os.path.join(
                            self.root, "VOCdevkit", "VOC2012", "SegmentationClass"
                        )
                    ),
                    os.path.exists(
                        os.path.join(
                            self.root, "VOCdevkit", "VOC2012", "SegmentationObject"
                        )
                    ),
                ]
            ):
                raise Exception(
                    f"""Dataset Not found / is of incorrrect fomat at {self.root}. Proper Format : 
                                /VOCdevkit
                                    --- /VOC2012
                                    ------ /Annotations
                                    ------ /JPEGImages
                                    ------ / ImageSets
                                    ------ /SegmentationClass
                                    ------ /SegmentationObject


                 """
                )
        self.dataset_dict = {}
        for item in self.split_types:
            dataset_type = item
            if item == "test":
                data = dataset(
                    root=self.root,
                    image_set="val",
                    download=self.download,
                    transform=self.transform[item],
                    target_transform=self.target_transform[item],
                )
            else:
                data = dataset(
                    root=self.root,
                    image_set="train",
                    download=self.download,
                    transform=self.transform[item],
                    target_transform=self.target_transform[item],
                )
            self.dataset_dict[dataset_type] = data

    def build_dict_info(self):
        """
        Behavior:
            This function creates info key in the output dictionary. The info key contains details related to the size
            of the training, validation and test datasets. Further, it can be used to define any additional information
            necessary for the user.
        Returns:
            dataset_dict (dict): Updated with info key that contains details related to the data splits
        """
        self.dataset_dict["info"] = {}
        self.dataset_dict["info"]["train_size"] = len(self.dataset_dict["train"])
        self.dataset_dict["info"]["val_size"] = len(self.dataset_dict["val"])
        self.dataset_dict["info"]["note"] = ""
        return self.dataset_dict
