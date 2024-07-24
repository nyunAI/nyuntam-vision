Here is the improved version of the documentation for adding new dataset formats:

### Adding New Dataset Formats
Users can use custom datasets to aid their compression job with nyuntam-vision by following the standard [dataset format](https://nyunai.github.io/nyun-docs/dataset/).

- **Dataset Folder Structure**
  To add a new dataset format, create a new file in the path mentioned below:
```
	   core
		.../data.py
	    .../dataset/
	    .../.../__init__.py
	    .../.../dataset.py
	    .../.../ {NEW_DATASET_FORMAT}.py
	 ```

- **Dataset Class Structure**
  A dataset class should be defined inside `{NEW_DATASET_FORMAT}.py`. The dataset class must inherit `BaseDataset` from `dataset.py`. `__init__` is required for execution. It must populate `self.dataset_dict`, which is a dictionary containing train, val, and test dataset objects. `build_dict_info` is another required function that defines meta-info for the `BaseDataset` class. Required meta-info can be found [here](https://github.com/nyunAI/nyuntam-vision/blob/main/core/datasets/cifar.py).

  ```python
  class Dataset(BaseDataset):
      def __init__(
          self,
          name=None, # Name of dataset
          root=None, # Root directory of folder
          transform=None, # Transform for images
          target_transform=None, # Transform for target
          download=True, # Download from the internet
          split_types=None, # [train, val] or [train, test, val]
          val_fraction=0.2, # Percentage of samples to be considered for validation
          shuffle=True, # Enables shuffling of data samples
          random_seed=None, # Sets random seed
      ):
          super(Dataset, self).__init__(
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

      def build_dict_info(self):
          # Implementation of the function
  ```

- **Adding to `__init__.py` for the Dataset**
  The `__init__.py` by default has the class `ClassificationDatasetFactory` which has a `create_dataset` function that is called internally to execute dataset loading. Update the `create_dataset` function to add the newly added dataset.

  ```python
  from .{NEW_DATASET_FORMAT} import Dataset

  class ClassificationDatasetFactory(object):
      @staticmethod
      def create_dataset(**kwargs):
          name = kwargs.get('name', None)
          ...
          if 'CIFAR10' == name:
              obj_dfactory = CIFAR10Dataset(**kwargs)
          elif '{NEW_DATASET_NAME}' == name:
              obj_dfactory = Dataset(**kwargs)
          else:
              obj_dfactory = CustomDataset(**kwargs)
          ...
  ```

Follow these steps to add a new dataset format to nyuntam-vison
