# import the necessary packages
import torch
import os
# define the base path to the input dataset and then use it to derive
# the path to the input images and annotation CSV files
BASE_PATH = "dataset"
IMAGES_PATH = 'train'
ANNOTS_PATH = "boxes_train.json"
ANNOTS_TEST_PATH = "boxes_test.json"

# define the path to the base output directory
BASE_OUTPUT = "output"
MODEL_PATH = os.path.sep.join([BASE_OUTPUT, "pyt_detector_3.pth"])

# determine the current device and based on that set the pin memory
# flag
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PIN_MEMORY = True if DEVICE == "cuda" else False