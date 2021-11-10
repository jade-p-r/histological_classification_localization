# runs using  python3 train.py -f hist
from sklearn.model_selection import StratifiedKFold
from classif_utils.utils import *
import pickle
import sys
import logging
import argparse
import os
from classif_utils import config
from classif_utils.models import Model
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))


def train(args, folds=5):
    """
    Training loop using k fold cross validation - we used k=5
    :return: none, trained model saved as pickle file locally under config filemae
    :rtype: none
    """
    train_dir = config.IMAGES_PATH
    classes = os.listdir(train_dir)
    train_df = pd.DataFrame()

    train_df = build_features(args['features'], train_df, classes, train_dir)
    ids_to_remove = duplicated_ids(train_dir)
    train_df = train_df.drop(train_df[train_df.image_name.isin(ids_to_remove)].index)
    y = target(train_df)
    scaled_X = preprocessing(train_df)

    k_folds = folds
    kfold = StratifiedKFold(n_splits=k_folds, shuffle=True)

    for train_index, val_index in kfold.split(scaled_X, y):
        X_train, X_val = scaled_X[train_index], scaled_X[val_index]
        y_train, y_val = y[train_index], y[val_index]
        clf = Model(model_type='svc')
        clf.fit(X_train, y_train)
        logger.debug(clf.score(clf.predict(X_val), y_val))
    filename = config.MODEL_PATH
    pickle.dump(clf, open(filename, 'wb'))


if __name__ == "__main__":
    args = argument_parser()
    train(args)
