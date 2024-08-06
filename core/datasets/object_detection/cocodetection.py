import torchvision
from torchvision.datasets.utils import download_and_extract_archive
from .dataset import BaseDataset
import os
from vision.core.utils.ultralyticsutils import create_ultralytics_folder_type
from pycocotools.coco import COCO
import yaml
import shutil


class CocoDetectionDataset(BaseDataset):
    """
    MSCOCO 2017

    Expected Directory Structure
    ----------------------------

     ------ train2017/ ---> Contains training images
     ------ val2017/   ---> Contains validation images
     ------ annotations/
     ----------- instances_train2017.json  ---> Train Annotations
     ----------- instances_val2017.json    ---> Val Annotations


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
        custom=False,
    ):
        super(CocoDetectionDataset, self).__init__(
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
        self.custom_root = os.path.join(self.root, "root")
        dataset = torchvision.datasets.CocoDetection
        if (
            not all(
                [
                    os.path.exists(os.path.join(self.custom_root, "val2017")),
                    os.path.exists(os.path.join(self.custom_root, "train2017")),
                    os.path.exists(
                        os.path.join(
                            self.custom_root, "annotations", "instances_val2017.json"
                        )
                    ),
                    os.path.exists(
                        os.path.join(
                            self.custom_root, "annotations", "instances_train2017.json"
                        )
                    ),
                ]
            )
            and self.download == False
        ):
            raise Exception(
                f"""Dataset Not found / is of incorrrect fomat at {self.custom_root}. Proper Format : 
                                //root
                                    //train
                                    //val
                                    //annotations
                                """
            )
        elif download == True:
            if not all(
                [
                    os.path.exists(os.path.join(self.custom_root, "val2017")),
                    os.path.exists(os.path.join(self.custom_root, "train2017")),
                    os.path.exists(
                        os.path.join(
                            self.custom_root, "annotations", "instances_val2017.json"
                        )
                    ),
                    os.path.exists(
                        os.path.join(
                            self.custom_root, "annotations", "instances_train2017.json"
                        )
                    ),
                ]
            ):
                self.download_mscoco(self.custom_root)
            else:
                pass
        else:
            raise Exception("Not Proper Format Provided for dataset")

        self.create_support_yaml()
        self.perform_ultra_format()

        self.dataset_dict = {}
        for item in self.split_types:
            dataset_type = item
            if item == "test":
                data = dataset(
                    root=os.path.join(self.custom_root, "val2017"),
                    annFile=os.path.join(
                        self.custom_root, "annotations", "instances_val2017.json"
                    ),
                    transform=self.transform[item],
                    target_transform=self.target_transform[item],
                )
            else:
                data = dataset(
                    root=os.path.join(self.custom_root, "train2017"),
                    annFile=os.path.join(
                        self.custom_root, "annotations", "instances_train2017.json"
                    ),
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

    def download_mscoco(self, to_path):
        if not all(
            [
                os.path.exists(os.path.join(self.custom_root, "val")),
                os.path.exists(os.path.join(self.custom_root, "train")),
                os.path.exists(
                    os.path.join(self.custom_root, "annotations", "train.json")
                ),
                os.path.exists(
                    os.path.join(self.custom_root, "annotations", "val.json")
                ),
            ]
        ):
            download_and_extract_archive(
                "http://images.cocodataset.org/zips/train2017.zip", to_path
            )
            download_and_extract_archive(
                "http://images.cocodataset.org/zips/val2017.zip", to_path
            )
            download_and_extract_archive(
                "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
                to_path,
            )

    def write_yaml_to_files(self, py_obj, filename):
        with open(
            f"{filename}.yaml",
            "w",
        ) as f:
            yaml.dump(py_obj, f, sort_keys=False)

    def perform_ultra_format(self):
        if os.path.exists(os.path.join(self.custom_root, "coco_format_data")):
            shutil.rmtree(os.path.join(self.custom_root, "coco_format_data"))
        create_ultralytics_folder_type(self.custom_root, "coco_format_data")

    def create_support_yaml(self):
        ann_path = os.path.join(
            self.custom_root, "annotations", "instances_val2017.json"
        )
        coco_annotation = COCO(annotation_file=ann_path)
        cat_ids = coco_annotation.getCatIds()

        cats = coco_annotation.loadCats(cat_ids)
        path = f"{self.custom_root}/coco_format_data"
        cat_names = [cat["name"] for cat in cats]
        class_dict = {i - 1: name for i, name in zip(cat_ids, cat_names)}
        yaml_config_obj = {
            "path": self.root,
            "train": f"{path}/images/train2017",
            "val": f"{path}/images/val2017",
            "nc": len(cat_ids),
            "names": class_dict,
        }
        self.write_yaml_to_files(
            yaml_config_obj, os.path.join(self.root, "supporting_yaml_coco_format")
        )
