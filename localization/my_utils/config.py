# import the necessary packages
import torch
import os
# define the base path to the input dataset and then use it to derive
# the path to the input images and annotation CSV files
BASE_PATH = "dataset"
IMAGES_PATH = 'train'
#IMAGES_PATH = os.path.sep.join([BASE_PATH, "train"])
ANNOTS_PATH = os.path.sep.join([BASE_PATH, "boxes_train.json"])
# define the path to the base output directory
BASE_OUTPUT = "output"
# define the path to the output model, label encoder, plots output
# directory, and testing utils paths
MODEL_PATH = os.path.sep.join([BASE_OUTPUT, "pyt_detector_3.pth"])
LE_PATH = os.path.sep.join([BASE_OUTPUT, "le.pickle"])
PLOTS_PATH = os.path.sep.join([BASE_OUTPUT, "plots"])
TEST_PATHS = os.path.sep.join([BASE_OUTPUT, "test_paths.txt"])

# determine the current device and based on that set the pin memory
# flag
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PIN_MEMORY = True if DEVICE == "cuda" else False
# specify mean and standard deviation for the set
MEAN = [0.688, 0.569, 0.776]
STD = [0.181, 0.231, 0.193]
# initialize our initial learning rate, number of epochs to train
# for, and the batch size
INIT_LR = 1e-4
NUM_EPOCHS = 2
BATCH_SIZE = 32
# specify the loss weights
LABELS = 1.0
BBOX = 1.0
