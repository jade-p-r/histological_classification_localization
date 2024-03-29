import pickle
from classif_utils.utils import *
import logging
import sys
from classif_utils import config
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))



def load_model(filename: np.ndarray):
    """
    loads model from a filename
    :param filename: filename
    :type filename:str
    :return:
    :rtype:
    """
    loaded_model = pickle.load(open(filename, 'rb'))
    return loaded_model


def im_class(nuclei):
    """
    returns the associated class of an image
    :param nuclei:
    :type nuclei:
    :return:
    :rtype:
    """
    if nuclei:
        return "nuclei"
    else:
        return "no_nuclei"


def plot_predictions(df: pd.DataFrame) -> None:
    """
    Plots the images from predetermined image name column within a given dataframe
    :param df:
    :type df:
    :return:
    :rtype:
    """

    for _, row in df.iterrows():
        im_name = row['image_name']
        im_path = os.path.join(config.TEST_PATH, im_class(row['nuclei']), im_name)
        cv2.imshow(im_name, cv2.imread(im_path))
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def predict():
    """
    prediction loop on the test set
    :return: prediction loop over directory of test images
    :rtype:
    """
    test_dir = config.TEST_PATH
    features_list = config.FEATURES

    classes = os.listdir(test_dir)
    test_df = pd.DataFrame()
    test_df = build_features(features_list, test_df, classes, test_dir)

    y = target(test_df)
    scaled_X = preprocessing(test_df)

    model = load_model(config.MODEL_PATH)
    y_pred = model.predict(scaled_X)

    test_df['predictions'] = pd.Series(y_pred)
    test_df.to_csv(config.PREDICTIONS_PATH)

    precision, recall, fn, fp, tn, tp = precision_recall(test_df)

    logger.debug("the precisions is {} and recall is {}".format(precision, recall))
    logger.debug(
        "FALSE POSITIVES : {} , FALSE NEGATIVES : {} , TRUE POSITIVES : {}, TRUE NEGATIVES : {}".format(fp.shape[0],
                                                                                                        fn.shape[0],
                                                                                                        tp.shape[0],
                                                                                                        tn.shape[0]))

    plot_predictions(fn)
    plot_predictions(fp)
    return


if __name__ == "__main__":
    predict()