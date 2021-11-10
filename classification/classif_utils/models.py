import pandas as pd
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
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
from sklearn.metrics import accuracy_score


class Model:
    """
    Model class that allows to train with different models
    """
    def __init__(self, model_type):
        if model_type == 'svc':
            self.model = SVC(kernel='linear')
        elif model_type == 'rf':
            self.model = RandomForestClassifier()
        else:
            self.model = LogisticRegression()

    def fit(self, X_train, y_train):
        self.model = self.model.fit(X_train, y_train)

    def predict(self, X):
        return self.model.predict(X)

    def score(self, y_pred, y_gt):
        return accuracy_score(y_pred, y_gt)


class ColorImageFeatures:
    """
    Class of color - 3 channels - image features related to image texture or content to train the model with
    """
    def __init__(self, image):
        self.image = image
        self.colors = ("blue", "green", "red")
        self.channel_ids = (0, 1, 2)

    def variance_of_laplacian(self) -> float:
        """
        Calculates the variance of the laplacian of the image, an indicator of blurriness in the image
        Some False Negatives were identified as blurry

        :return: laplacian of the input image
        :rtype: float
        """
        return cv2.Laplacian(self.image, cv2.CV_64F).var()

    def fft_std(self) -> float:
        """
        Calculates the std - indicator of variance - of the Fast Fourier Transform of the image on one of the axis characterizing blank or nearly blank images
        Some FN were identified as blank or nearly blank
        :return: fft std
        :rtype:float
        """
        test = np.fft.fft2(self.image[:, :, 2])
        ftest = np.fft.fftshift(test)
        fft_std = np.std(ftest[:, 100])
        return fft_std

    def count_nuclei_watershed(self):
        """
        Calculated two methods - one being the filter of the other - of the number of nuclei of the image using watershed segmentation
        :return: number of nuclei
        :rtype: int
        """

        imgray = self.image[:, :, 0]
        histogram, bin_edges = np.histogram(imgray, bins=256, range=(0, 1))

        for channel_id, c in zip(self.channel_ids, self.colors):
            histogram, bin_edges = np.histogram(
                self.image[:, :, channel_id], bins=256, range=(0, 256)
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

        markers = cv2.watershed(self.image, markers)
        self.image[markers == -1] = [255, 0, 0]
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

    def count_nuclei_stains(self) -> float:
        """
        Calculated the number of nuclei of the image using stain image and clustering segmentation
        :return: number of nuclei
        :rtype: int
        """
        # get nuclei/hematoxylin channel
        im_nuclei_stain = self.stain_im()[:, :, 0]

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

    def differences(self):
        """
        Calculates the difference between two channels of the image
        :return: Two single channel images being  differences between channels / colors of the image
        :rtype: array, array
        """
        diff1 = self.image[:, :, 1] - self.image[:, :, 0]
        diff2 = self.image[:, :, 2] - self.image[:, :, 0]
        return diff1, diff2

    def stain_im(self, mean_ref, std_ref, W):
        """
        Returns the transform of the image in the stains space using HistomicsTK library considering an image of reference
        :param mean_ref: reference image mean
        :type mean_ref: float
        :param std_ref: reference umage std
        :type std_ref: float
        :param W: stain matrix
        :type W: bytearray
        :return: stain image
        :rtype: array
        """
        im_nmzd = htk.preprocessing.color_normalization.reinhard(self.image, mean_ref, std_ref)
        im_stains = htk.preprocessing.color_deconvolution.color_deconvolution(im_nmzd, W).Stains
        return im_stains

    def histogram(self, channel_id):
        """
        Calculates the histogram of the image on the given channel
        :return: histogram
        :rtype:bytearray
        """
        histogram, bin_edges = np.histogram(
            self.image[:, :, channel_id], bins=256, range=(0, 256)
        )
        return histogram

    def dhash(self, hashSize=8):
        """
        Computes the hash value for a given image
        :param hashSize: size of the hash
        :type hashSize:int
        :return:hash
        :rtype:int
        """
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (hashSize + 1, hashSize))
        diff = resized[:, 1:] > resized[:, :-1]
        return sum([2 ** i for (i, v) in enumerate(diff.flatten()) if v])

    def training_features(self, features_list, features_dict):
        """
        Training features calculation given a features list . Appends a dictionary with selected featuresa dn return the dictionary
        :param features_list: features list
        :type features_list: list
        :param features_dict: features dict
        :type features_dict: dict
        :return: features dictionary
        :rtype: dict
        """
        if 'hist' in features_list:
            for channel_id, c in zip(self.channel_ids, self.colors):
                features_dict['hist_' + str(c)] = self.histogram(channel_id)
        if 'laplacian' in features_list:
            features_dict['laplacian'] = self.variance_of_laplacian()
        if 'watershed' in features_list:
            count1, count2 = self.count_nuclei_watershed()
            features_dict['watershed_1'] = count1
            features_dict['watershed_2'] = count2
        if 'stain_count' in features_list:
            features_dict['stain'] = self.count_nuclei_stains()
        return features_dict

