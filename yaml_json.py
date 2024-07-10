import yaml
import json
import os
import copy
import shutil
import zipfile 
import tarfile

def get_recipe_filename(method,framework,model_name):
    # This function replaces training_recipies_dic
    if method == "CPU Post Training Quantization - Torch" and framework in ['timm','torchvision','huggingface']:
        return "FXQuant_classification.yaml"
    elif method == "CPU Quant Aware Training - Torch" and framework in ['timm','torchvision','huggingface']:
        return "FXQuant_qat_classification.yaml"
    elif method == "CPU Post Training Quantization - OpenVino" and framework in ['timm','torchvision','huggingface','mmdet']:
        if framework == 'mmdet':
            return "NNCFQuant_MMDet.yaml"
        else:
            return "NNCFQuant_classification.yaml"
    elif method == "CPU Post Training Quantization - ONNX" and framework in ['timm','torchvision','huggingface']:
        return "ONNXQuant_classification.yaml"
    elif method == "GPU Post Training Quantization - TensorRT"and framework in ['timm','torchvision','huggingface','mmdet','mmyolo']:
        if framework =='mmdet':
            return "TRT_MMDet.yaml"
        elif framework  == 'mmyolo':
            return "TRT_MMYolo.yaml"
        else:
            return "TRTQuant_classification.yaml"
    elif method == "CPU Quantization Aware Training - OpenVino" and framework in ['timm','torchvision','huggingface']:
        return "NNCFQAT.yaml"
    elif method == "Structural Pruning" and framework in ['timm','torchvision','huggingface','mmdet','mmyolo']:
        if framework =='mmdet':
            return "MMRazor_MMDet.yaml"
        elif framework == 'mmyolo':
            return "MMRazorPrune_MMYolo.yaml"
        else:
            if any(typ in model_name for typ in ["vit","deit","swin"]):
                return "vit_pruning.yaml"
            else:
                return "classification_pruning.yaml"
    elif method == "Distillation"  and framework in ['timm','torchvision','huggingface','mmdet','mmyolo']:
        if framework == 'mmdet':
            return "MMRazorDistill_MMDet.yaml"
        elif framework == 'mmyolo':
            return  "MMRazorDistill_MMYolo.yaml"
        else:
            return "KDTransfer_Classification.yaml"
    elif method == "LLM Quantization" and framework in ['huggingface']:
        return "AWQ_llm.yaml"
    elif method == "LLM Structured Pruning" and framework in ['huggingface']:
        return "FLAP_llm.yaml"
    elif method == "LLM Engine ExLlama" and framework in ['huggingface']:
        return "ExLlama_llm.yaml"
    elif method == "LLM Engine TensorRT" and framework in ['huggingface']:
        return "TensorRT_llm.yaml"
    elif method == "LLM Retraining-free Structured Pruning" and framework in ['huggingface']:
        return "FLAP_llm.yaml"#deprecated
    elif method == "LLM Engine MLCLLM" and framework in ['huggingface']:
        return "MLC_llm.yaml"
    else:
        raise ValueError(f"The requested Compression Algorithm {method} does not support {framework} models")
    
frameworks_dic ={
    'Timm': 'timm',
    'HuggingFace':'huggingface',
    'Torchvision': 'torchvision',
    'MM Detection': 'mmdet',
    'MM Segmentation':'mmseg' 
}
tasks_dic={
 'Image Classification': 'image_classification',
 'Object Detection': 'object_detection',
 'Image Segmentation': 'segmentation',
 'LLM': "llm"
}

dataset_name_dic={
    'Object Detection - Coco':'COCODETECTION',
    'Object Detection - VOC':'VOCDETECTION',
    'Image Segmentation - VOC': 'VOCSEGMENTATION',
    'LLM - Alpaca' : 'finetune_alpaca', # to be deprecated (not used)
    'LLM - Single Column': 'llm_single_column',
    'LLM - Multi Columns': 'llm_multi_column',
    'Image Classification': 'image_classification',
}
def fill_data(training_recipe, json_data):
    yaml_recipe = copy.deepcopy(training_recipe)
    yaml_recipe["USER_FOLDER"] = "/workspace/Kompress/user_data"
    yaml_recipe["JOB_ID"] = str(json_data["id"])
    yaml_recipe["JOB_SERVICE"] = str(json_data["job_service"])
    yaml_recipe["MODEL"] = json_data["model"]["model_subname"]
    yaml_recipe["PLATFORM"] = frameworks_dic[json_data["model"]["framework"]]
    yaml_recipe["TASK"] = tasks_dic[json_data["task"]]
    yaml_recipe['DATASET_ID'] = json_data["dataset"]["id"]
    try:
        yaml_recipe["DATASET_NAME"] = dataset_name_dic[json_data["dataset"]["dataset_name"]]
    except:
        yaml_recipe["DATASET_NAME"] = json_data["dataset"]["dataset_name"]

    if json_data['dataset']['dataset_relative_folder_path'] != '':
        if yaml_recipe['TASK'] == "image_classification":
            yaml_recipe['DATASET_NAME'] ='custom'
        elif yaml_recipe['TASK'] == 'object_detection':
            yaml_recipe['DATASET_NAME'] = "COCODETECTION"
        elif yaml_recipe['TASK'] == 'segmentation':
            yaml_recipe['DATASET_NAME'] == 'VOCSEGMENTATION'
        elif yaml_recipe['TASK'] == "llm":
            pass
        else:
            raise ValueError("This Task doesnt support custom datasets")

    yaml_recipe["DATA_PATH"] = os.path.join(
        yaml_recipe["USER_FOLDER"], f"datasets/{yaml_recipe['DATASET_ID']}"
    )
    yaml_recipe["LOGGING_PATH"] = os.path.join(yaml_recipe["USER_FOLDER"], f"logs/{yaml_recipe['JOB_SERVICE']}/{yaml_recipe['JOB_ID']}")

    yaml_recipe["CACHE_PATH"] = os.path.join(yaml_recipe["USER_FOLDER"], ".cache")

    yaml_recipe["JOB_PATH"] = os.path.join(
        yaml_recipe["USER_FOLDER"], f"jobs/{yaml_recipe['JOB_SERVICE']}/{yaml_recipe['JOB_ID']}"
    )
    yaml_recipe["MODEL_PATH"] = os.path.join(
        yaml_recipe["USER_FOLDER"], f"jobs/{yaml_recipe['JOB_SERVICE']}/{yaml_recipe['JOB_ID']}"
    )
    if json_data['model']['model_relative_folder_path'] != "":
        yaml_recipe["CUSTOM_MODEL_PATH"] = os.path.join(
            yaml_recipe["USER_FOLDER"], f'models/{json_data["model"]["id"]}'
        )
    else:
        yaml_recipe["CUSTOM_MODEL_PATH"] = ""
    hyperparameters = json_data["method_hyperparameters"]
    if "" != hyperparameters:
        hyper_keys = list(hyperparameters.keys())
        yaml_keys = list(yaml_recipe.keys())
        yaml_keys.remove(yaml_recipe["ALGO_TYPE"])
        yaml_hyper_keys = list(
            yaml_recipe[yaml_recipe["ALGO_TYPE"]][yaml_recipe["ALGORITHM"]].keys()
        )
        for k in yaml_keys:
            if k in hyper_keys:
                yaml_recipe[k] = hyperparameters[k]
        for k in yaml_hyper_keys:
            if k in hyper_keys:
                yaml_recipe[yaml_recipe["ALGO_TYPE"]][yaml_recipe["ALGORITHM"]][k] = hyperparameters[k]

    if json_data['dataset']['dataset_relative_folder_path'] !="" and os.path.exists(os.path.join('/workspace/Kompress/custom_data',json_data['dataset']['dataset_relative_folder_path'])) :
        original_data_location = os.path.join('/workspace/Kompress/custom_data',json_data['dataset']['dataset_relative_folder_path'])
    else:
        original_data_location = None
    if json_data['model']['model_relative_folder_path'] != "" and os.path.exists(os.path.join('/workspace/Kompress/custom_data',json_data['model']['model_relative_folder_path'])):
        original_model_location = os.path.join('/workspace/Kompress/custom_data',json_data['model']['model_relative_folder_path'])
    else:
        original_model_location =None
    return yaml_recipe , original_data_location, original_model_location


def execute_yaml_creation(json_path):
    f = open(json_path)
    json_data = json.load(f)

    method_name = json_data["method"]
    model_name = json_data['model']['model_name']
    framework_name = frameworks_dic[json_data['model']['framework']]
    yaml_file = get_recipe_filename(method_name,framework_name,model_name)
    with open(os.path.join("training_recipies", yaml_file)) as file:
        training_recipe = yaml.full_load(file)
    yaml_recipe, original_data_locaiton, original_model_location = fill_data(training_recipe, json_data)

    # Creating all the folders 
    create_folders(yaml_recipe , original_data_locaiton , original_model_location)
    #Recurring Issue - The line gets added again due to improper merging before commits
    #shutil.copy(json_path, os.path.join(yaml_recipe["JOB_PATH"], "config.json"))
    with open(os.path.join(yaml_recipe["JOB_PATH"], "config.yaml"), "w") as file:
        documents = yaml.dump(yaml_recipe, file)

    return os.path.join(yaml_recipe["JOB_PATH"], "config.yaml")


def create_folders(yaml_recipe , original_data_location = None , original_model_location = None):
    folder_name = yaml_recipe["USER_FOLDER"]
    if not os.path.exists(f"{folder_name}"):
        os.makedirs(f"{folder_name}", exist_ok=True)
        os.makedirs(f"{folder_name}/models", exist_ok=True)
        os.makedirs(f"{folder_name}/datasets", exist_ok=True)
        os.makedirs(f"{folder_name}/jobs", exist_ok=True)
        os.makedirs(f"{folder_name}/logs", exist_ok=True)
        os.makedirs(f"{folder_name}/.cache", exist_ok=True)
    os.makedirs(f"{folder_name}/datasets/{yaml_recipe['DATASET_ID']}", exist_ok=True)  

    if yaml_recipe['CUSTOM_MODEL_PATH'] and yaml_recipe['CUSTOM_MODEL_PATH'] != '':
        os.makedirs(yaml_recipe['CUSTOM_MODEL_PATH'], exist_ok=True)
    os.makedirs(f"{folder_name}/jobs/{yaml_recipe['JOB_SERVICE']}/{yaml_recipe['JOB_ID']}", exist_ok=True)
    os.makedirs(f"{folder_name}/logs/{yaml_recipe['JOB_SERVICE']}/{yaml_recipe['JOB_ID']}", exist_ok=True)
    #moving dataset
    if original_data_location:
        move_files(yaml_recipe,original_data_location , f"{folder_name}/datasets/{yaml_recipe['DATASET_ID']}" , mode = "data")
    #moving model
    if original_model_location and yaml_recipe['CUSTOM_MODEL_PATH'] != '':
        move_files(yaml_recipe,original_model_location , yaml_recipe["CUSTOM_MODEL_PATH"] , mode = "model")
    
def move_files(yaml_recipe,original_location, new_location , mode):
    if mode == "data":
        shutil.copytree(original_location , new_location, dirs_exist_ok=True)
    elif mode=="model":
        if len(os.listdir(original_location))==1:
            file = os.listdir(original_location)[0]
            extension = file.split('.')[-1]
            if extension not in ['zip' , 'tar' , 'gz', 'bz2']:
                new_path = os.path.join(new_location , f'wds.{extension}')
                shutil.copy(os.path.join(original_location,file) , new_path)
            
            elif extension == 'zip':
                os.makedirs(new_location, exist_ok=True)
                with zipfile.ZipFile(os.path.join(original_location,file) , 'r') as zip_ref:
                    
                    zip_ref.extractall(new_location)
                    contents = os.listdir(new_location)
                    for folder in contents:
                        folder_path = os.path.join(new_location , folder)
                        for files in os.listdir(folder_path):
                            tmp_file_path = os.path.join(folder_path , files)
                            shutil.move(tmp_file_path , os.path.join(folder_path , ".."))
                shutil.rmtree(folder_path)
            elif extension in ['tar' , 'gz' , 'bz2']:
                os.makedirs(new_location, exist_ok=True)
                with tarfile.TarFile(original_location , 'r') as tar_ref:
                    tar_ref.extractall(new_location)
                    contents = os.listdir(new_location)
                    for folder in contents:
                        folder_path = os.path.join(new_location , folder)
                        for files in os.listdir(folder_path):
                            tmp_file_path = os.path.join(folder_path , files)
                            shutil.move(tmp_file_path , os.path.join(folder_path , ".."))
                shutil.rmtree(folder_path)
        else:
            shutil.copytree(original_location , new_location, dirs_exist_ok=True)          
        
                
            
            
            
    


if __name__ == "__main__":
    json_path = "sample.json"
    execute_yaml_creation(json_path)
