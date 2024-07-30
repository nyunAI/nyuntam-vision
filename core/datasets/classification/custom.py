from .dataset import BaseDataset
from PIL import Image
import os
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
import gdown
import zipfile
import glob


class Custom(Dataset):
    """
    Custom Dataset
    Expected Format
    Custom
        train
            c1
              images
            c2
              images

        val
            c1
                images
            c2
                images
    References
    ----------
    CheXpert: A Large Chest Radiograph Dataset with Uncertainty Labels and Expert Comparison,
    Irvin et al, 2019.

    Parameters
    ----------
        name (string): dataset name 'CHEST', default=None.
        root (string): Root directory where ``train, train.csv`` exists or will be saved if download flag is set to
        True (default is None).
        mode (string): Mode to run the dataset, options are 'train', 'val', 'test', 'heatmap', default='train'.
        subname (string): Subname of the dataset, default='atelectasis'. options are 'atelectasis', 'edema', 'cardiomegaly', 'effusion'
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set, default=None.
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``. Default=None.
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it, default=None.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again, default=True.
        split_types (list): the possible values of this parameter are "train", "test" and "val".
            If the split_type contains "val", then shuffle has to be True, default value is None.
        val_fraction (float): If float, should be between 0.0 and 1.0 and represent
        the proportion of the dataset to include in the val split.
        shuffle (bool): Whether or not to shuffle the data before splitting into val from train,
            default is True. If shuffle is true, there should be 'val' in split_types.
        random_seed (int): RandomState instance, default=None.
    """

    def __init__(
        self,
        root=None,
        transform=None,
        target_transform=None,
        mode="train",
        download=True,
    ):
        self._labels = []
        self._mode = mode
        self.transform = transform
        self.target_transform = target_transform
        self._image_paths = glob.glob(f"{root}/{self._mode}/*/*")

        self._label_names = [
            l.split("/")[-1] for l in glob.glob(f"{root}/{self._mode}/*")
        ]
        self._label_to_number = {}
        self._number_to_label = {}
        for i, ln in enumerate(self._label_names):
            self._label_to_number[str(ln)] = i
            self._label_to_number[str(i)] = str(ln)
        self._labels = [
            self._label_to_number[str(ln)]
            for ln in [l.split("/")[-2] for l in self._image_paths]
        ]
        self._num_image = len(self._image_paths)

    def __len__(self):
        return self._num_image

    def __getitem__(self, idx):
        image = Image.open(self._image_paths[idx]).convert("RGB")

        if self.transform != None:
            image = self.transform(image)

        label = torch.tensor([self._labels[idx]]).float()

        if self._mode == "train" or self._mode == "val" or self._mode == "test":

            return [torch.tensor(image), label]
        else:
            raise Exception("Unknown mode : {}".format(self._mode))


class CustomDataset(BaseDataset):
    """

    Parameters
    ----------
        name (string): dataset name 'CHEST', default=None.
        root (string): Root directory where ``train, train.csv`` exists or will be saved if download flag is set to
        True (default is None).
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set, default=None.
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``. Default=None.
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it, default=None.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again, default=True.
        split_types (list): the possible values of this parameter are "train", "test" and "val".
            If the split_type contains "val", then shuffle has to be True, default value is None.
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
        download=False,
        split_types=None,
        val_fraction=0.2,
        shuffle=True,
        random_seed=None,
    ):
        super(CustomDataset, self).__init__(
            name=name,
            root=os.path.join(root, "root"),
            transform=transform,
            target_transform=target_transform,
            download=download,
            split_types=split_types,
            val_fraction=val_fraction,
            shuffle=shuffle,
            random_seed=random_seed,
        )

        dataset = Custom
        self.dataset_dict = {}

        for item in self.split_types:
            dataset_type = item
            if item == "val" and not self.val_exists:
                self.dataset_dict[dataset_type] = None
                continue
            data = dataset(
                root=self.root,
                mode=item,
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

        return self.dataset_dict
