

### Adding New Dataset Formats
Users can use custom datasets to aid thier compression job with nyuntam-vision by following the standard [dataset format](https://nyunai.github.io/nyun-docs/dataset/)
- **Dataset Folder Structure**
			To add a new dataset format you need to create a new file in the path mentioned below. 
	```
	   core
		.../data.py
	    .../dataset/
	    .../.../__init__.py
	    .../.../dataset.py
	    .../.../ {NEW_DATASET_FORMAT}.py
	 ```
- **Dataset Class Structure**
	A dataset class should be defined inside ``{NEW_DATASET_FORMAT}.py``. The dataset class must inherit ``BaseDataset`` from ``dataset.py``. ``\_\_init\_\_`` and is required for execution. It must populate ``self.dataset_dict`` which is dictionary containing train,val and test dataset objects. ``build_dict_info`` is another required function that defines metainfo for the ``BaseDataset`` class. Required metainfo can be found [here](https://github.com/nyunAI/nyuntam-vision/blob/main/core/datasets/cifar.py).
	```python
	class Dataset(BaseDataset):
	    def __init__(
	        self,
	        name=None, # name of dataset
	        root=None,# Root Directory of folder
	        transform=None, #Transform for Images
	        target_transform=None, #Transform for Target
	        download=True, # Download from the internet
	        split_types=None, #[train,val] or [train,test,val]
	        val_fraction=0.2, # Percentage of Samples to be considered for validation
	        shuffle=True, #Enables shuffling of Datasamples
	        random_seed=None,# Sets Random Seed
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
	        
	 ```
- **Adding to \_\_init\_\_.py for the Dataset**
	The ``\_\_init\_\_.py`` by default has the class ``ClassificatinDatasetFactory`` which has a ``create_dataset`` function that is called internally to execute dataset loading. Update the ``create_dataset`` function to add the newly added dataset.
	```python
		#Import the Algorithm class previously defined
		from {
		class ClassificationDatasetFactory(object):
			@staticmethod
		    def create_dataset(**kwargs):
		    ...
		    if 'CIFAR10' == name:
	            obj_dfactory = CIFAR10Dataset(**kwargs)
	        elif 'DATASET' == name:
		        obj_dfactory = Dataset(**kwargs)
	        else:
	            obj_dfactory = CustomDataset(**kwargs)
        ...
        ...
     
	```
-