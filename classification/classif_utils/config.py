# import the necessary packages
import torch
import os
# define the base path to the input dataset and then use it to derive
# the path to the input images and annotation CSV files
IMAGES_PATH = 'train'
TEST_PATH = "test"
ANNOTS_PATH = "boxes_train.json"
ANNOTS_TEST_PATH = "boxes_test.json"
COLORS = ("blue", "green", "red")
CHANNEL_IDS = (0, 1, 2)
# define the path to the base output directory
BASE_OUTPUT = "output"
MODEL_PATH = os.path.sep.join([BASE_OUTPUT, "classification_model.sav"])
PREDICTIONS_PATH = os.path.sep.join([BASE_OUTPUT, "predictions.csv"])
REF_IM = 'train_nuclei_45.png'