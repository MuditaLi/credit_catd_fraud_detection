# -*- coding: utf-8 -*-
import os
from pathlib import Path

# Root path
ROOT_NAME = 'Credit_card_fraud_detect'
PATH_REPOSITORY_ROOT = Path('.').resolve()

# folder names
DIRNAME_DOCS = 'docs'
DIRNAME_DATA = 'data'
DIRNAME_SRC = 'src'
DIRNAME_DATA_RAW = 'raw'
DIRNAME_DATA_PROCESSED = 'processed'
DIRNAME_DATA_EXTERNAL = 'external'
DIRNAME_DATA_OUTPUT = 'output'
DIRNAME_CONFIG_FILES = 'configs'
DIRNAME_MODELS = 'models'

# first level path
PATH_DOCS = os.path.realpath(os.path.join(PATH_REPOSITORY_ROOT, DIRNAME_DOCS))
PATH_DATA = os.path.realpath(os.path.join(PATH_REPOSITORY_ROOT, DIRNAME_DATA))
PATH_SRC = os.path.realpath(os.path.join(PATH_REPOSITORY_ROOT, DIRNAME_SRC))

# Path to data folder
PATH_DATA_RAW = os.path.realpath(os.path.join(PATH_DATA, DIRNAME_DATA_RAW))
PATH_DATA_PROCESSED = os.path.realpath(os.path.join(PATH_DATA, DIRNAME_DATA_PROCESSED))
PATH_DATA_EXTERNAL = os.path.realpath(os.path.join(PATH_DATA, DIRNAME_DATA_EXTERNAL))
PATH_DATA_OUTPUT = os.path.realpath(os.path.join(PATH_DATA, DIRNAME_DATA_OUTPUT))
# Path to configs file folder
PATH_CONFIG = os.path.realpath(os.path.join(PATH_SRC, DIRNAME_CONFIG_FILES))
# Path to models folder
PATH_MODELS = os.path.realpath(os.path.join(PATH_SRC, DIRNAME_MODELS))

PATH_CURRENT_REPOSITORY = os.path.dirname(os.path.realpath(__file__))

