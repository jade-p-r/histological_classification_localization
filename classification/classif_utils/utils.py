import cv2
import os
import matplotlib.pyplot as plt
import histomicstk as htk
import numpy as np
from tqdm import tqdm
import pandas as pd
from sklearn.preprocessing import StandardScaler
import scipy as sp
import skimage.io
import skimage.measure
import skimage.color
from typing import List
from .models import ColorImageFeatures
from . import config

# create stain to color map
stainColorMap = {
    'hematoxylin': [0.65, 0.70, 0.29],
    'eosin':       [0.07, 0.99, 0.11],
    'dab':         [0.27, 0.57, 0.78],
    'null':        [0.0, 0.0, 0.0]
}

# specify stains of input image
stain_1 = 'hematoxylin'   # nuclei stain
stain_2 = 'eosin'         # cytoplasm stain
stain_3 = 'null'          # set to null of input contains only two stains

# create stain matrix
W = np.array([stainColorMap[stain_1],
              stainColorMap[stain_2],
              stainColorMap[stain_3]]).T


def load_images(image_dir: str, train_dir: str) -> List[float]:
    """
    Loads images from a given directory following train/test images structure
    :param image_dir: path to image dir
    :type image_dir: str
    :return: list of image hashes within directory
    :rtype: List[float]
    """
    dhashes = {}
    for sub_dir in os.listdir(image_dir):
        image_list = os.listdir(os.path.join(train_dir, sub_dir))

        for image in tqdm(image_list):
            image_path = os.path.join(image_dir, sub_dir, image)
            im = cv2.imread(image_path)
            dhashes[image] = ColorImageFeatures().dhash(im)
    return dhashes


def define_ref_im_attributes(dir, classes, image_name):
    """
    Defines the coefficient for color normalization based on an image of reference
    :param dir:
    :type dir:
    :param classes:
    :type classes:
    :param image_name:
    :type image_name:
    :return:
    :rtype:
    """
    ref_im_path = os.path.join(dir, classes[1], image_name)
    ref_im = cv2.imread(ref_im_path)
    mean_ref, std_ref = htk.preprocessing.color_conversion.lab_mean_std(ref_im)
    return mean_ref, std_ref


def build_features(features_list: List, df: pd.DataFrame, classes: List, dir: str):
    """
    Adds all metrics to the input dataframe via a dictionary. Features include histograms,
    nuclei counts using different methods and Fast Fourier Transforms attributes
    :param features_list:
    :type features_list:
    :param df:
    :type df:
    :param classes:
    :type classes:
    :param dir:
    :type dir:
    :param W:
    :type W:
    :return:
    :rtype:
    """
    for class_ in classes:
        image_list = os.listdir(os.path.join(dir, class_))
        image_dic = {'nuclei': not 'no' in class_}
        for image in tqdm(image_list):
            image_dic['image_name'] = image
            image_path = os.path.join(dir, class_, image)
            im = cv2.imread(image_path)
            im_features = ColorImageFeatures(im)
            im_features_dic = im_features.training_features(features_list, image_dic)
            df = df.append(im_features_dic, ignore_index=True)

    return df


def select_correlated_features(df: pd.DataFrame) -> List[int]:
    """
    Selects the correlated features of a given dataframe
    :param df: dataframe
    :type df: pd.dataframe
    :return: list of indexes to drop
    :rtype: List
    """
    df = df.drop('nuclei')
    corr_matrix = df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
    to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
    return to_drop


def preprocessing(df: pd.DataFrame) -> np.ndarray:
    """
    Completes the preprocessing of a features dataframe to return standardized features ready for training
    :param df: dataframe of numerical features
    :type df: dataframe
    :return: scaled and reshaped array of features
    :rtype: array
    """
    hist_cols = [col for col in df.columns if 'hist_' in col]
    X = df[hist_cols]

    flat_X = [item for sublist in X.values for item in sublist]
    flat_X = np.asarray(flat_X)
    flat_X = flat_X.reshape(X.shape[0], np.shape(flat_X)[1] * X.shape[1])
    scaler = StandardScaler()
    flat_X = scaler.fit_transform(flat_X)
    return flat_X


def duplicated_ids(directory: str) -> List[str]:
    """
    Selects duplicates ids of an dataframe by computes hash values of the images
    :param directory: path to image directory
    :type directory:str
    :return: list of duplicated ids
    :rtype:bytearray
    """
    train_hashes = load_images(directory, config.IMAGES_PATH)
    dictB = {}
    for key, value in train_hashes.items():
        dictB.setdefault(value, set()).add(key)
    res = list(filter(lambda x: len(x) > 1, dictB.values()))
    ids_to_remove = [list(im_dict)[0] for im_dict in res]
    return ids_to_remove


def target(df: pd.DataFrame) -> np.ndarray:
    """
    Selects target variable array and convert to float for training
    :param df: dataframe for prediction
    :type df:dataframe
    :return: y array for target
    :rtype: array
    """

    df.nuclei = df.nuclei.astype(int)
    y = df['nuclei']
    y = np.asarray(y).astype('float32')
    return y


def precision_recall(df):
    """
    Computes preicison metrics for evaluation of precision and recall and identification of false postivies and false negatives
    :param df: dataframe with predictions
    :type df: dataframe
    :return: precision, recall and each sub dataframe for each categroy
    :rtype:arrays
    """
    fp = df.loc[(df['nuclei'] == 0) & (df.predictions == 1.0)]
    fn = df.loc[(df['nuclei'] == 1) & (df.predictions == 0.0)]
    tp = df.loc[(df['nuclei'] == 1) & (df.predictions == 1.0)]
    tn = df.loc[(df['nuclei'] == 0) & (df.predictions == 0.0)]

    precision = tp.shape[0] / (tp.shape[0] + fp.shape[0])
    recall = tp.shape[0] / (tp.shape[0] + fn.shape[0])
    return precision, recall, fn, fp, tn, tp
