from sklearn import svm
from sklearn.model_selection import StratifiedKFold
from utils import *
import pickle
import sys
from typing import List
import logging

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

train_dir = "train/"
colors = ("blue", "green", "red")
channel_ids = (0, 1, 2)


def dhash(image, hashSize=8):
    """

    :param image:
    :type image:
    :param hashSize:
    :type hashSize:
    :return:
    :rtype:
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (hashSize + 1, hashSize))
    diff = resized[:, 1:] > resized[:, :-1]
    return sum([2 ** i for (i, v) in enumerate(diff.flatten()) if v])


def load_images(image_dir: str) -> List[float]:
    """

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
            dhashes[image] = dhash(im)
    return dhashes


def duplicated_ids(directory: str) -> List[str]:
    """

    :param directory:
    :type directory:
    :return: list of duplicated ids
    :rtype:
    """
    train_hashes = load_images(directory)
    dictB = {}
    for key, value in train_hashes.items():
        dictB.setdefault(value, set()).add(key)
    res = list(filter(lambda x: len(x) > 1, dictB.values()))
    ids_to_remove = [list(im_dict)[0] for im_dict in res]
    return ids_to_remove

def train():
    """

    :return: none, trained model saved as pickle file locally under config filemae
    :rtype: none
    """
    classes = os.listdir(train_dir)
    train_im_list = os.listdir(os.path.join(train_dir, classes[0]))
    train_df = pd.DataFrame()
    train_dic = {}

    train_df = add_metrics(train_df, classes, train_dir, W)
    #correlated_cols = select_correlated_features(train_df)
    #train_df = train_df.drop(correlated_cols, axis=1, inplace=True)
    ids_to_remove = duplicated_ids(train_dir)
    train_df = train_df.drop(train_df[train_df.image_name.isin(ids_to_remove)].index)
    y = target(train_df)
    scaled_X = preprocessing(train_df)

    k_folds = 5
    kfold = StratifiedKFold(n_splits=k_folds, shuffle=True)

    for train_index, val_index in kfold.split(scaled_X, y):
        X_train, X_val = scaled_X[train_index], scaled_X[val_index]
        y_train, y_val = y[train_index], y[val_index]
        clf = svm.SVC(kernel='linear')
        clf = clf.fit(X_train, y_train)
        logger.debug(clf.score(X_val, y_val))
    filename = 'first_model.sav'
    pickle.dump(clf, open(filename, 'wb'))


if __name__ == "__main__":
    train()
