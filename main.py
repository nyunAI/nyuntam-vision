# print('Running Main File')
DEBUG=False
import sys
import os
from yaml_json import execute_yaml_creation
sys.path.append(os.path.abspath(os.path.join(".")))
sys.path.append(os.path.abspath(os.path.join(".", "Adapt")))
sys.path.append(os.path.abspath(os.path.join(".", "algorithms_kompress", "llm", "tensorrtllm")))
sys.path.append(os.path.abspath(os.path.join(".", "algorithms_kompress", "llm", "prune", "FLAP")))
from factory import CompressionFactory
import yaml
import argparse
from version import __version__

from logging_kompress import define_logger
import logging


parser = argparse.ArgumentParser()
parser.add_argument("--yaml_path", default="", type=str)
parser.add_argument("--json_path", default="sample.json", type=str)


args = parser.parse_args()
if args.yaml_path == "":
    yaml_path = execute_yaml_creation(args.json_path)
    file = yaml_path
else:
    file = args.yaml_path

with open(file, "r") as f:
    arg = yaml.safe_load(f)

logging.info(f"Kompress version: {__version__}")
if DEBUG ==False:
    try:
        factory = CompressionFactory(**arg)
        name = factory()
        logger = logging.getLogger(name)
        logger.info("JOB COMPLETED")

    except Exception as e:
        logger = define_logger("failure" , arg.get("LOGGING_PATH"))
        logger.exception(e)
        logger.info("JOB FAILED")
else:
    factory = CompressionFactory(**arg)
    name = factory()
    logger = logging.getLogger(name)
