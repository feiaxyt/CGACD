# Copyright (c) SenseTime. All Rights Reserved.

import os
import logging
import math
import sys

logs = set()

def get_format(logger, level):
    format_str = '[%(asctime)s-%(filename)s#%(lineno)3d] %(message)s'
    formatter = logging.Formatter(format_str)
    return formatter

def init_log(name, level=logging.INFO, format_func=get_format):
    if (name, level) in logs:
        return
    logs.add((name, level))
    logger = logging.getLogger(name)
    logger.setLevel(level)
    ch = logging.StreamHandler()
    ch.setLevel(level)
    formatter = format_func(logger, level)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger

def add_file_handler(name, log_file, level=logging.INFO):
    logger = logging.getLogger(name)
    fh = logging.FileHandler(log_file)
    fh.setFormatter(get_format(logger, level))
    logger.addHandler(fh)

init_log('global')