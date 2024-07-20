

### Adding New Algorithms
Adding New Compression algorithms is simple and highly flexible. New algorithms can be added for any of the supported compression type and adding support for new compression types can be added by mimicking the folder structure of existing compression types. This page provides the steps to add new algorithms to the existing compression stack.
 - **Algorithms Structure**:
	Nyuntam-Vision currently supports three compression types - quant, prune and distill. The folder structure for all types of compression algorithms is same. Adding a new Compression type can be achieved by mimicking the below specified folder structure. 
     ```
   CompressionType (quant,prune,distill)
    .../__init__.py
    .../AlgorithmFolder #Eg. NNCF
    .../.../__init__.py
    .../.../main.py # contains the algorithm class
    .../.../auxilary files 
   ```
 - **Algorithm Class Structure**
	Inside the ``main.py`` present in ``CompressionType/AlgorithmFolder`` a class containing two compulsory functions must be defined. This class is responsible for compressing models using the newly added compression algorithm. The required functions are \_\_init\_\_() and compress_model(). 
	- \_\_init\_\_ provides the the following parameters to the class
		 -  model -> The model to be compressed
		 - loaders -> The dictionary containing train/val dataloaders 
		 - kwargs -> All the hyperparameters from yaml config
	- compress_model() is the function which is called internally to compress the model. The function must return the compressed model and the \_\_name\_\_ variable. 


	```python
	  class Algorithm:
		  def __init__(self, model, loaders=None, **kwargs):
			  self.model = model #The model to be compressed (after loading custom weights)
			  self.loaders = loaders # a dictionary containing train and val torch dataloaders
			  self.batch_size = getattr(kwargs,"BATCH_SIZE") #Example for retrieving hyperparameters from kwargs
		  def compress_model():
			  #The main function that executes the compression task.
			  #This function is called from factory.py and does not contain parameters
			  return compressed_model, __name__
	  ```
- **Adding to \_\_init\_\_.py for the Algorithm**
	The algorithm must be called in `CompressionType/AlgorithmFolder/\_\_init\_\_.py`` and the imported classes and modules must be added to \_\_all\_\_. The developer may chose to import additional modules and classes in addition to the Algorithm Class as needed.
	```python
	#Import the Algorithm class previously defined
	from .main import Algorithm
	#Add it to the __all__
	__all__ = ["Algorithm"]

	```
- **Adding to \_\_init\_\_.py of quant/prune/distill**
	The \_\_init\_\_.py  present in ``CompressionType/AlgorithmFolder`` contains ``initialize_initialization`` function that conditionally initialize the required packages for execution of the implemented compression algorithm. Follow the below example for adding your algorithm to the initialize_initialization() function.
	```python
	def initialize_initialization(algoname):
    if algoname == 'FXQuant':
        from .TorchNativeQuantization import FXQuant
        return FXQuant
    ...
    #Add initialization of the defined Algorithm
    elif algoname == "{ALGORITHM_NAME}":
        # Import Algorithm from the AlgorithmFolder and return it
	    from .AlgorithmFolder import Algorithm
	    return Algorithm 
	    # This setup is done to prevent unneeded imports and possible circular imports
    else:
        return None
	```
- **Constructing YAML for  the newly added algorithm**
Follow the below steps to construct the yaml for specifying the setting for the compression. 
- ```yaml
	-	# Specify the algorithm name as used in initialize_initalization()
		ALGORITHM : {ALGORITHM_NAME} 
		#Specify the Compression_Type Folder Name (default support: {quant,prune,distill})
		ALGO_TYPE: {COMPRESSION_TYPE}
		#Specify other needed general parameters
		
		{COMPRESSION_TYPE}:
			{ALGORITHM_NAME}:
				#Algorithm Specific Hyperparameters
	```
	The ``ALGORITHM`` parameter represents your Algorithm name as used in ``initialize_initalization`` function.  The ``ALGO_TYPE`` parameter represents the Compression_Type (quant,distill,prune).
**NOTE:** If a custom compression type is utillized it ``ALGO_TYPE`` parameter must contain the exact folder name of the custom compression type.

These are the steps required to add a new algorithm to the existing compression stack.