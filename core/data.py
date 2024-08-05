from torchvision.datasets.utils import download_and_extract_archive
from torchvision.datasets import ImageFolder
from torchvision import transforms
from .datasets import (
    ObjectDetectionDatasetFactory,
    ClassificationDatasetFactory,
    SegmentationDatasetFactory,
    PoseEstimationDatasetFactory,
)
import os


def prepare_data(dataset_name: str, url: str, to_path: str, **kwargs):
    """Prepare provided data link for model training/compression.
    The file type and possible file compression is automatically detected from the provided file link.
    Args:
        url (str): URL to the dataset, the URL must contain a single parent folder or zip containing a single parent folder.
                    The directory structure of the parent folder should be as follow :
                    - data (parent folder)
                        - train
                            - class1
                                - img1
                                - img2
                                - img3
                            - class2
                        - val
                            - class1
                                - img1
                                - img2
                            - class2
                        - test
                            - class1
                                - img1
                                - img2
                            - class2
        to_path (str): Path to the directory the data will be extracted to.
    Returns:
        datasets (tuple) : A tuple of train, val and test pytorch Datasets
    """
    insize = kwargs.get("insize", 32)
    if url:
        download_and_extract_archive(url, to_path)
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        transform = transforms.Compose(
            [transforms.Resize(insize), transforms.ToTensor(), normalize]
        )
        train_dataset = ImageFolder(os.path.join(to_path, "data/train"), transform)
        val_dataset = ImageFolder(os.path.join(to_path, "data/val"), transform)
        test_dataset = ImageFolder(os.path.join(to_path, "data/test"), transform)
        dataset_dict = {
            "train": train_dataset,
            "val": val_dataset,
            "test": test_dataset,
        }
    else:
        imsize = kwargs.get("insize", 32)
        transform = transforms.Compose(
            [transforms.Resize((imsize, imsize)), transforms.ToTensor()]
        )

        transforms1 = {"train": transform, "val": transform, "test": transform}
        target_transforms = {"train": None, "val": None, "test": None}
        task = kwargs.get("TASK", "image_classification")
        if task == "image_classification":
            dataset_dict = ClassificationDatasetFactory.create_dataset(
                name=dataset_name,
                root=to_path,
                split_types=["train", "val", "test"],
                val_fraction=0.2,
                transform=transforms1,
                target_transform=target_transforms,
            )
        elif task == "object_detection":
            dataset_dict = ObjectDetectionDatasetFactory.create_dataset(
                name=dataset_name,
                root=to_path,
                split_types=["train", "val", "test"],
                val_fraction=0.2,
                transform=transforms1,
                target_transform=target_transforms,
            )
        elif task == "segmentation":
            dataset_dict = SegmentationDatasetFactory.create_dataset(
                name=dataset_name,
                root=to_path,
                split_types=["train", "val", "test"],
                val_fraction=0.2,
                transform=transforms1,
                target_transform=target_transforms,
            )
        elif task == "pose_estimation":
            dataset_dict = PoseEstimationDatasetFactory.create_dataset(
                name=dataset_name,
                root=to_path,
                split_types=["train", "val", "test"],
                val_fraction=0.2,
                transform=transforms1,
                target_transform=target_transforms,
            )
        else:
            raise Exception(
                f"Unknown Task {task}. Known Tasks =[image_classification,object_detection,segmentation,tracking,pose_estimation]"
            )
    return dataset_dict
