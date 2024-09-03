  

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

## Acknowledgments and Citations

This repository utilizes various state-of-the-art methods and algorithms developed by the research community. We acknowledge the following works that have contributed to the development and performance of the Nyuntam suite:

- **OpenMMLab Model Compression Toolbox and Benchmark**  
  *MMRazor Contributors.* 2021. [GitHub Repository](https://github.com/open-mmlab/mmrazor)

  ```bibtex
  @misc{2021mmrazor,
    title={OpenMMLab Model Compression Toolbox and Benchmark},
    author={MMRazor Contributors},
    howpublished = {\url{https://github.com/open-mmlab/mmrazor}},
    year={2021}
  }
  ```

- **Neural network compression framework for fast model inference**  
  *Kozlov Alexander, Lazarevich Ivan, Shamporov Vasily, Lyalyushkin Nikolay, Gorbachev Yury.* arXiv preprint, 2020.

  ```bibtex
  @article{kozlov2020neural,
    title =   {Neural network compression framework for fast model inference},
    author =  {Kozlov Alexander and Lazarevich Ivan and Shamporov Vasily and Lyalyushkin Nikolay and Gorbachev Yury},
    journal = {arXiv preprint arXiv:2002.08679},
    year =    {2020}
  }
  ```

- **MMDetection: Open MMLab Detection Toolbox and Benchmark**  
  *Chen Kai, Wang Jiaqi, Pang Jiangmiao, Cao Yuhang, Xiong Yu, Li Xiaoxiao, Sun Shuyang, et al.* arXiv preprint, 2019.

  ```bibtex
  @article{mmdetection,
    title   = {{MMDetection}: Open MMLab Detection Toolbox and Benchmark},
    author  = {Chen Kai and Wang Jiaqi and Pang Jiangmiao and Cao Yuhang and Xiong Yu and Li Xiaoxiao and Sun Shuyang and Feng Wansen and Liu Ziwei and Xu Jiarui and Zhang Zheng and Cheng Dazhi and Zhu Chenchen and Cheng Tianheng and Zhao Qijie and Li Buyu and Lu Xin and Zhu Rui and Wu Yue and Dai Jifeng and Wang Jingdong and Shi Jianping and Ouyang Wanli and Loy Chen Change and Lin Dahua},
    journal= {arXiv preprint arXiv:1906.07155},
    year={2019}
  }
  ```

- **MMYOLO: OpenMMLab YOLO Series Toolbox and Benchmark**  
  *MMYOLO Contributors.* 2022. [GitHub Repository](https://github.com/open-mmlab/mmyolo)

  ```bibtex
  @misc{mmyolo2022,
    title={{MMYOLO: OpenMMLab YOLO} series toolbox and benchmark},
    author={MMYOLO Contributors},
    howpublished = {\url{https://github.com/open-mmlab/mmyolo}},
    year={2022}
  }
  ```

- **MMSegmentation: OpenMMLab Semantic Segmentation Toolbox and Benchmark**  
  *MMSegmentation Contributors.* 2020. [GitHub Repository](https://github.com/open-mmlab/mmsegmentation)

  ```bibtex
  @misc{mmseg2020,
    title={{MMSegmentation}: OpenMMLab Semantic Segmentation Toolbox and Benchmark},
    author={MMSegmentation Contributors},
    howpublished = {\url{https://github.com/open-mmlab/mmsegmentation}},
    year={2020}
  }
  ```

- **OpenMMLab Pose Estimation Toolbox and Benchmark**  
  *MMPose Contributors.* 2020. [GitHub Repository](https://github.com/open-mmlab/mmpose)

  ```bibtex
  @misc{mmpose2020,
    title={OpenMMLab Pose Estimation Toolbox and Benchmark},
    author={MMPose Contributors},
    howpublished = {\url{https://github.com/open-mmlab/mmpose}},
    year={2020}
  }
  ```

- **A White Paper on Neural Network Quantization**  
  *Nagel Markus, Fournarakis Marios, Amjad RanaAli, Bondarenko Yelysei, Baalen Martvan, Blankevoort Tijmen.* Cornell University - arXiv, 2021.

  ```bibtex
  @misc{Nagel_Fournarakis_Amjad_Bondarenko_Baalen_Blankevoort_2021,
    title={A White Paper on Neural Network Quantization},
    journal={Cornell University - arXiv},
    author={Nagel Markus and Fournarakis Marios and Amjad RanaAli and Bondarenko Yelysei and Baalen Martvan and Blankevoort Tijmen},
    year={2021},
    month={Jun}
  }
  ```

- **Learned Step Size Quantization**  
  *Esser StevenK., McKinstry JeffreyL., Bablani Deepika, Appuswamy Rathinakumar, Modha DharmendraS.* arXiv: Learning, 2019.

  ```bibtex
  @misc{Esser_McKinstry_Bablani_Appuswamy_Modha_2019,
    title={Learned Step Size Quantization},
    journal={arXiv: Learning},
    author={Esser StevenK. and McKinstry JeffreyL. and Bablani Deepika and Appuswamy Rathinakumar and Modha DharmendraS.},
    year={2019},
    month={Feb}
  }
  ```

- **Group Fisher Pruning for Practical Network Compression**  
  *Liu Liyang, Zhang Shilong, Kuang Zhanghui, Zhou Aojun, Xue Jing-hao, Wang Xinjiang, Chen Yimin, et al.* Proceedings of the 38th International Conference on Machine Learning, 2021.

  ```bibtex
  @InProceedings{Liu:2021,
    TITLE      = {Group Fisher Pruning for Practical Network Compression},
    AUTHOR     = {Liu Liyang AND Zhang Shilong AND Kuang Zhanghui AND Zhou Aojun AND Xue Jing-hao AND Wang Xinjiang AND Chen Yimin AND Yang Wenming AND Liao Qingmin AND Zhang Wayne},
    BOOKTITLE  = {Proceedings of the 38th International Conference on Machine Learning},
    YEAR       = {2021},
    SERIES     = {Proceedings of Machine Learning Research},
    MONTH      = {18--24 Jul},
    PUBLISHER  = {PMLR}
  }
  ```

- **Channel-Wise Knowledge Distillation for Dense Prediction**  
  *Shu Changyong, Liu Yifan, Gao Jianfei, Yan Zheng, Shen Chunhua.* Proceedings of the IEEE/CVF International Conference on Computer Vision, 2021.

  ```bibtex
  @inproceedings{shu2021channel,
    title={Channel-Wise Knowledge Distillation for Dense Prediction},
    author={Shu Changyong and Liu Yifan and Gao Jianfei and Yan Zheng and Shen Chunhua},
    booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
    pages={5311--5320},
    year={2021}
  }
  ```

- **PKD: General Distillation Framework for Object Detectors via Pearson Correlation Coefficient**  
  *Cao Weihan, Zhang Yifan, Gao Jianfei, Cheng Anda, Cheng Ke, Cheng Jian.* arXiv preprint, 2022.

  ```bibtex
  @article{cao2022pkd,
    title={PKD: General Distillation Framework for Object Detectors via Pearson Correlation Coefficient},
    author={Cao Weihan and Zhang Yifan and Gao Jianfei and Cheng Anda and Cheng Ke and Cheng Jian},
    journal={arXiv preprint arXiv:2207.02039},
    year={2022}
  }
  ```

- **Depgraph: Towards any structural pruning**  
  *Fang Gongfan, Ma Xinyin, Song Mingli, Mi Michael Bi, Wang Xinchao.* Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023.

  ```bibtex
  @inproceedings{fang2023depgraph,
    title={Depgraph: Towards any structural pruning},
    author={Fang Gongfan and Ma Xinyin and Song Mingli and Mi Michael Bi and Wang Xinchao},
    booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
    pages={16091--16101},
    year={2023}
  }
  ```
