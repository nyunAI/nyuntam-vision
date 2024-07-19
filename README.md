# Kompress Vision
Developed to compress and optimize deep learning models, Kompress provides a set of compression techniques tailored for specific deployment constraints. Users have the flexibility to choose and combine multiple techniques to achieve the best trade-off between model performance and deployment constraints. Leveraging cutting-edge techniques like pruning, quantization, distillation, etc., Kompress achieves exceptional model compression levels on a variety of language and vision models.

## INSTALLATION
Installation can be performed either via installing requirements in a virtual environment or by utilizing our docker images. To quickly run Kompress for experimentation and usage, utilize Nyun CLI to get Kompress running in a few clicks. For contributing to Kompress  build docker containers from the available docker image or create a virtual enviroment from the provided requirements.txt.
### NYUN_CLI
To Install and use Nyun CLI head over to  the Nyun CLI Documentation through the link below.
[Nyun-CLI](https://github.com/nyunAI/nyunzero-cli)

### Setting up via Docker 
Using our pre-made docker image allows the users to setup Nyun Kompress without the hassle of installing the requirements. It can also be utilized as a development environment.
1. **Git clone** 
	Clone the Nyuntam Repository - the base repository used for executing Nyuntam Vision repository
   ```bash
   git clone https://github.com/nyunAI/nyuntam.git --recursive
   cd Nyuntam
   ```
2. **Docker pull**
  Pull our pre-made docker image from dockerhub. ``docker pull`` can take a few minutes to complete download and requires a stable internet . 
   ```bash
   docker pull nyunadmin/nyunzero_kompress_vision:v0.1
   ```
3. **Docker run**
Create a Docker Container from the downloaded image using the  `` docker run`` command.
   ```bash 
   docker run -it -d --gpus all -v /dev/shm:/dev/shm -v $(pwd):/workspace --name {CONTAINER_NAME} --network=host yunadmin/nyunzero_kompress_vision:v0.1 bash 
   ```
   <b>NOTE:</b> nvidia-container-toolkit is expected to be installed before the execution of this
4. **Docker exec**
  Enter into the created docker container using the ``docker exec`` command. <i>Tip: Use cntl + D to exit the container.  A docker cheat sheet can be found <a href ="">here</a> </i>
   ```bash
   docker exec -it {CONTAINER_NAME} /bin/bash
   ```

### Setting Up via Virtual Environments
This method requires creating a virtual environment and installing the requirements through the provided requirements.txt. 
1. Clone the Nyuntam Repository - the base repository used for executing Nyuntam Vision repository
   ```bash
   git clone https://github.com/nyunAI/nyuntam.git --recursive
   cd nyuntam
   ```
3. Create a virtual environment using  either Anaconda or Venv
   ```bash
   python3 -m venv {ENVIRONMENT_NAME}
   source {ENVIRONMENT_NAME}/bin/activate
   ```
   or
    ```bash
   conda create -n {ENVIRONMENT_NAME}
   conda activate {ENVIRONMENT_NAME}
   ```  
4. **Pip install requirements**
   ```bash
   pip install -r requirements.txt
   ```


## USAGE
### Simple Usage (Existing Dataset)
- **Sample YAML / JSON** 
 We follow a yaml format for specifying the hyperparameters.  The example below shows the basic yaml format. Complete YAML files for each Algorithm Can be found <a href=''>here</a> and details of supported hyperparameters can be found <a href=''>here</a>
  ```yaml
  # Sample YAML file
  #General Parameters
  DATASET_NAME: CIFAR10 
  MODEL: resnet50
  JOB_PATH: 'ABC/jobs/1'
  ...
  ...
  #Type of Compression
  quant:
	  # Compression Algo Specific Params
	  EPOCHS: 1
	  OPTIMIZER: 'Adam'
	 ...
	 ...
	  
  ```
   **YAML explanation**: The general-commonly used hyperparameters are specified first. We then have a sub section to specify Compression Algorithm specific parameters.
 - **Details of IO paths**
   ```
   {USER_FOLDER}
    .../jobs
	....../mds.xx   # Output generated from the compression task
	.../datasets
	....../DatasetA # Folder containing datasets
	.../models
	....../wds.xx   # Input model Weight file
	.../logs
	....../log.log  # Log file which records job history
   ```
	 - Corresponding Paths in YAML:
	       
	```yaml
	  JOB_PATH: /{USER_FOLDER}/jobs
	  DATASET_PATH: /{USER_FOLDER}/datasets
	  LOGGING_PATH: /{USER_FOLDER}/logs
	  ```
- **Running a Sample YAML file**
	```bash
	#Enter Nyuntam Repository
	cd Nyuntam
	#Run main.py file and provide the yaml configuration
	python main.py --yaml_path <yaml_path>
	```
 

### Custom Model / Dataset
- **Details of custom dataset path**
	  Custom datasets can be added at ``{USER_FOLDER}/datasets`` and must follow the standard format for custom datasets which can be found in the [docs](https://nyunai.github.io/nyun-docs/dataset/). Other formats can be supported by extending the ``BaseDataset`` class. 
- **Details of custom model weight path**
  Custom Weights can be added to {USER_FOLDER}/models and be named ``wds.pt`` or ``wds.pth``. The ``CUSTOM_MODEL_PATH`` hyperparameter must be updated in the yaml to point to the weight file.
  


## DEV GUIDE
### (How to Add New Algorithms

### How to Add New Models

### How to Add New Dataset

### Documentation link
[nyun docs](https://github.com/nyun-docs)
