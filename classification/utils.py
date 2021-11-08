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


colors = ("blue", "green", "red")
channel_ids = (0, 1, 2)
ref_image = 'train_nuclei_45.png'
train_dir = "train/"
test_dir = "test/"
classes = os.listdir(test_dir)

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

ref_im_path = os.path.join(train_dir, classes[1], ref_image)
ref_im = cv2.imread(ref_im_path)
index = 24
train_nuc_list = os.listdir(os.path.join(train_dir, classes[1]))
im_input = cv2.imread(os.path.join(train_dir, classes[1], train_nuc_list[index]))
# get mean and stddev of reference image in lab space
mean_ref, std_ref = htk.preprocessing.color_conversion.lab_mean_std(ref_im)

def add_hist(im, image_dic, im_name):
    """

    :param im:
    :type im:
    :param image_dic:
    :type image_dic:
    :param im_name:
    :type im_name:
    :return: dictionary of features with added histogram
    :rtype:
    """
    for channel_id, c in zip(channel_ids, colors):
        histogram, bin_edges = np.histogram(
            im[:, :, channel_id], bins=256, range=(0, 256)
        )
        image_dic['hist_' + str(im_name) + str(c)] = histogram
    return image_dic


def add_single_hist(im, image_dic, im_name):
    histogram, bin_edges = np.histogram(
        im, bins=256, range=(0, 256)
    )
    image_dic['hist_' + str(im_name)] = histogram
    return image_dic


def add_metrics(df, classes, dir, W):
    for class_ in classes:
        image_list = os.listdir(os.path.join(dir, class_))
        image_dic = {'nuclei': not 'no' in class_}
        for image in tqdm(image_list):
            image_dic['image_name'] = image
            image_path = os.path.join(dir, class_, image)
            im = cv2.imread(image_path)
            diff1, diff2 = differences(im)
            im_nmzd = htk.preprocessing.color_normalization.reinhard(im, mean_ref, std_ref)
            im_stains = htk.preprocessing.color_deconvolution.color_deconvolution(im_nmzd, W).Stains
            image_dic = add_hist(im_stains, image_dic, "stains")
            image_dic = add_hist(im, image_dic, "orig")

            image_dic = add_single_hist(diff1, image_dic, "diff1")
            image_dic = add_single_hist(diff2, image_dic, "diff2")

            num_nuclei_stains = count_nuclei_stains(im_stains)
            num_nuclei_watershed = count_nuclei_watershed(im)
            image_dic['num_nuclei_watershed'] = num_nuclei_watershed
            image_dic['num_nuclei_stains'] = num_nuclei_stains


            im_fft_std = add_fft_std(im)
            image_dic['fft_std'] = im_fft_std


            im_laplacian_variance = variance_of_laplacian(im)
            image_dic['laplacian'] = im_laplacian_variance

            #image_dic['image'] = im
            df = df.append(image_dic, ignore_index=True)

    return df


def select_correlated_features(df: pd.DataFrame) -> List[int]:
    df = df.drop('nuclei')
    corr_matrix = df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
    to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
    return to_drop


def add_fft_std(im: np.ndarray) -> float:
    test = np.fft.fft2(im[:, :, 2])
    ftest = np.fft.fftshift(test)
    fft_std = np.std(ftest[:, 100])
    return fft_std


def count_nuclei_stains(im_stains: np.ndarray) -> float:
    # get nuclei/hematoxylin channel
    im_nuclei_stain = im_stains[:, :, 0]

    # segment foreground
    foreground_threshold = 20

    im_fgnd_mask = sp.ndimage.morphology.binary_fill_holes(
        im_nuclei_stain < foreground_threshold)

    # run adaptive multi-scale LoG filter
    min_radius = 1
    max_radius = 3

    im_log_max, im_sigma_max = htk.filters.shape.cdog(
        im_nuclei_stain, im_fgnd_mask,
        sigma_min=min_radius * np.sqrt(2),
        sigma_max=max_radius * np.sqrt(2)
    )

    # detect and segment nuclei using local maximum clustering
    local_max_search_radius = 20

    im_nuclei_seg_mask, seeds, maxima = htk.segmentation.nuclear.max_clustering(
        im_log_max, im_fgnd_mask, local_max_search_radius)

    # filter out small objects
    min_nucleus_area = 6

    im_nuclei_seg_mask = htk.segmentation.label.area_open(
        im_nuclei_seg_mask, min_nucleus_area).astype(np.int)

    # compute nuclei properties
    objProps = skimage.measure.regionprops(im_nuclei_seg_mask)
    return len(objProps)


def count_nuclei_watershed(im):

    imgray = im[:, :, 0]
    histogram, bin_edges = np.histogram(imgray, bins=256, range=(0, 1))

    for channel_id, c in zip(channel_ids, colors):
        histogram, bin_edges = np.histogram(
            im[:, :, channel_id], bins=256, range=(0, 256)
        )
        plt.plot(bin_edges[0:-1], histogram, color=c)

    plt.xlabel("Color value")
    plt.ylabel("Pixels")
    ret, thresh = cv2.threshold(imgray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # noise removal
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    # sure background area
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    # Finding sure foreground area
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    ret, sure_fg = cv2.threshold(dist_transform, 0.3 * dist_transform.max(), 255, 0)

    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)

    # Marker labelling
    ret, markers = cv2.connectedComponents(sure_fg)

    # Add one to all labels so that sure background is not 0, but 1
    markers = markers + 10

    # Now, mark the region of unknown with zero
    markers[unknown == 255] = 0

    markers = cv2.watershed(im, markers)
    im[markers == -1] = [255, 0, 0]
    markers1 = markers.astype(np.uint8)
    ret, m2 = cv2.threshold(markers1, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    contours, hierarchy = cv2.findContours(m2, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    count = []
    for cont in contours:
        area = cv2.contourArea(cont)
        x, y, w, h = cv2.boundingRect(cont)
        if w / h < 5 and h / w < 5 and area > 8 and area < 200:
            count.append(cont)
    return len(count), len(contours)


def differences(im):
    """

    :param im:
    :type im:
    :return: differences between channels / colors of the image
    :rtype:
    """
    diff1 = im[:, :, 1] - im[:, :, 0]
    diff2 = im[:, :, 2] - im[:, :, 0]
    return diff1, diff2


def variance_of_laplacian(image: np.ndarray) -> float:
    """

    :param image: input image
    :type image: image array BGR
    :return: laplacian of the input image
    :rtype: float
    """
    return cv2.Laplacian(image, cv2.CV_64F).var()


def preprocessing(df: pd.DataFrame) -> np.ndarray:
    """

    :param df: dataframe of numerical features
    :type df:
    :return: scaled and reshaped array of features
    :rtype:
    """
    hist_cols = [col for col in df.columns if 'hist_' in col]
    X = df[hist_cols]

    flat_X = [item for sublist in X.values for item in sublist]
    flat_X = np.asarray(flat_X)
    flat_X = flat_X.reshape(X.shape[0], np.shape(flat_X)[1] * X.shape[1])
    scaler = StandardScaler()
    flat_X = scaler.fit_transform(flat_X)
    return flat_X


def target(df: pd.DataFrame) -> np.ndarray:
    """

    :param df: dataframe for prediction
    :type df:
    :return: y array for target
    :rtype:
    """

    df.nuclei = df.nuclei.astype(int)
    y = df['nuclei']
    y = np.asarray(y).astype('float32')
    return y


def precision_recall(df):
    """

    :param df: dataframe with predictions
    :type df:
    :return: precision, recall and each sub dataframe for each categroy
    :rtype:
    """
    fp = df.loc[(df['nuclei'] == 0) & (df.predictions == 1.0)]
    fn = df.loc[(df['nuclei'] == 1) & (df.predictions == 0.0)]
    tp = df.loc[(df['nuclei'] == 1) & (df.predictions == 1.0)]
    tn = df.loc[(df['nuclei'] == 0) & (df.predictions == 0.0)]

    precision = tp.shape[0] / (tp.shape[0] + fp.shape[0])
    recall = tp.shape[0] / (tp.shape[0] + fn.shape[0])
    return precision, recall, fn, fp, tn, tp
