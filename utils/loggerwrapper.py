import logging
import os
from datetime import datetime


def get_logger():
    logger = logging.getLogger('mylogger')
    logger.setLevel(logging.INFO)

    os.makedirs("logs", exist_ok=True)
    fh = logging.FileHandler(f"logs/{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}.log")
    fh.setLevel(logging.INFO)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger


GLOBAL_LOGGER = get_logger()
