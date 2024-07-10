import logging
import os
import shutil
from datetime import datetime


def define_logger(name, logging_path):
    
    # clears handlers defined by previous logger(s)
    logging.root.handlers.clear()
    
    logging.basicConfig(
        filename=f"{logging_path}/log.log",
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H-%M-%S",
        level=logging.INFO,
    )

    logger = logging.getLogger(name)
    return logger