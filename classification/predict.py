import pickle
from classif_utils.utils import *
import logging
import sys
from classif_utils import config
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

filename = "first_model.sav"


def load_model(filename: np.ndarray):
    """

    :param filename:
    :type filename:
    :return:
    :rtype:
    """
    loaded_model = pickle.load(open(filename, 'rb'))
    return loaded_model


def im_class(nuclei):
    """

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

    :param df:
    :type df:
    :return:
    :rtype:
    """

    for _, row in df.iterrows():
        im_name = row['image_name']
        im_path = os.path.join(test_dir, im_class(row['nuclei']), im_name)
        cv2.imshow(im_name, cv2.imread(im_path))
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def predict():
    """

    :return: prediction loop over directory of test images
    :rtype:
    """
    test_dir = config.TEST_PATH
    classes = os.listdir(test_dir)
    test_df = pd.DataFrame()
    test_df = add_metrics(test_df, classes, test_dir, W)

    y = target(test_df)
    scaled_X = preprocessing(test_df)

    model = load_model(filename)
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