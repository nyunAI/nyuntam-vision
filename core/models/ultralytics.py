import torch
from ultralytics import YOLO
from ultralytics import RTDETR
import os


def get_YOLOV8(name, pretrained, cache_dir):
    present_dir = os.getcwd()
    os.chdir(cache_dir)
    model = YOLO(f"{name}.pt")
    os.chdir(present_dir)
    return model


def get_RTDETR(name, pretrained, cache_dir):
    present_dir = os.getcwd()
    os.chdir(cache_dir)
    model = RTDETR(f"{name}.pt")
    print(model)
    os.chdir(present_dir)
    return model
