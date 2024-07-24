  

# nyuntam-vision

Developed to compress and optimize deep learning models, Nyuntam Vision provides a set of compression techniques tailored for specific deployment constraints. Users have the flexibility to choose and combine multiple techniques to achieve the best trade-off between model performance and deployment constraints. Leveraging cutting-edge techniques like pruning, quantization, distillation, etc., Nyuntam Vision achieves exceptional model compression levels on a variety of language and vision models.

  

## Installation

Installation can be performed either by installing requirements in a virtual environment or by utilizing our docker images. To quickly run nyuntam-vision for experimentation and usage, utilize the Nyun CLI to get nyuntam-vision running in a few clicks. For contributing to nyuntam-vision, build docker containers from the available docker image or create a virtual environment from the provided `requirements.txt`.

  

### NYUN CLI

To install and use Nyun CLI, head over to the Nyun CLI documentation through the link below:

[Nyun-CLI](https://github.com/nyunAI/nyunzero-cli)

  

### Setting up via Docker

Using our pre-made Docker image allows users to set up Nyuntam Vision without the hassle of installing the requirements. It can also be utilized as a development environment.

  

1.  **Git clone**

Clone the Nyuntam Repository - the base repository used for executing the Nyuntam Vision repository:

```bash

git clone https://github.com/nyunAI/nyuntam.git --recursive

cd nyuntam

```

  

2.  **Docker pull**

Pull our pre-made Docker image from DockerHub. The `docker pull` command can take a few minutes to complete and requires a stable internet connection:

```bash

docker pull nyunadmin/nyunzero_kompress_vision:v0.1

```

  

3.  **Docker run**

Create a Docker container from the downloaded image using the `docker run` command:

```bash

docker run -it -d --gpus all -v /dev/shm:/dev/shm -v $(pwd):/workspace --name {CONTAINER_NAME}  --network=host  nyunadmin/nyunzero_kompress_vision:v0.1  bash

```

  

### Configuration YAML

Details about the YAML configuration can be found [here](#).

  

```yaml
# Sample YAML file
# General Parameters
DATASET_NAME: CIFAR10
MODEL: resnet50
JOB_PATH: 'ABC/jobs/1'
...
# Type of Compression
quant:
# Compression Algo Specific Params
EPOCHS: 1
OPTIMIZER: 'Adam'
...
```

  

**YAML explanation**: The general-commonly used hyperparameters are specified first. We then have a subsection to specify Compression Algorithm specific parameters.

  

-  **Details of IO paths**

```bash

{USER_FOLDER}

.../jobs

....../mds.xx # Output generated from the compression task

.../datasets

....../DatasetA # Folder containing datasets

.../models

....../wds.xx # Input model Weight file

.../logs

....../log.log # Log file which records job history

```

  

- Corresponding Paths in YAML:

```yaml

JOB_PATH: /{USER_FOLDER}/jobs

DATASET_PATH: /{USER_FOLDER}/datasets

LOGGING_PATH: /{USER_FOLDER}/logs

```

  

-  **Running a Sample YAML file**

```bash

# Enter Nyuntam Repository

cd nyuntam

# Run main.py file and provide the yaml configuration

python main.py --yaml_path <yaml_path>

```

  

### Utilizing Custom Model / Dataset for Compression

-  **Details of custom dataset path**

Custom datasets can be added at `{USER_FOLDER}/datasets` and must follow the standard format for custom datasets, which can be found in the [docs](https://nyunai.github.io/nyun-docs/dataset/). Other formats can be supported by extending the `BaseDataset` class.

  

-  **Details of custom model weight path**

Custom weights can be added to `{USER_FOLDER}/models` and named `wds.pt` or `wds.pth`. The `CUSTOM_MODEL_PATH` hyperparameter must be updated in the YAML to point to the weight file.

  

## DEV GUIDE

- [How to Add New Algorithms](development-guide/algorithm.md)

- [How to Add New Dataset Formats](development-guide/datasetformat.md)

  

For complete documentation of nyuntam-vision, visit [nyun docs](https://github.com/nyun-docs).
