from my_utils import config
from torchvision import transforms
import argparse
import torch
import cv2
import json
import os
import numpy as np
import transforms as T
import logging
import pandas as pd
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm
from my_utils import config
import sys

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))


transforms = transforms.Compose([
	transforms.ToPILImage(),
	transforms.ToTensor(),
])


def np_vec_no_jit_iou(bboxes1, bboxes2):
	"""

	:param bboxes1:
	:type bboxes1:
	:param bboxes2:
	:type bboxes2:
	:return: value of iou for each pair
	:rtype: array of iou
	"""
	x11, y11, x12, y12 = np.split(bboxes1, 4, axis=1)
	x21, y21, x22, y22 = np.split(bboxes2, 4, axis=1)
	xA = np.maximum(x11, np.transpose(x21))
	yA = np.maximum(y11, np.transpose(y21))
	xB = np.minimum(x12, np.transpose(x22))
	yB = np.minimum(y12, np.transpose(y22))
	interArea = np.maximum((xB - xA + 1), 0) * np.maximum((yB - yA + 1), 0)
	boxAArea = (x12 - x11 + 1) * (y12 - y11 + 1)
	boxBArea = (x22 - x21 + 1) * (y22 - y21 + 1)
	iou = interArea / (boxAArea + np.transpose(boxBArea) - interArea)
	return iou


def preprocess(image: np.ndarray) -> np.ndarray:
	"""

	:param image:
	:type image:
	:return: preprocessed image
	:rtype:
	"""
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	image = image.transpose((2, 0, 1))
	image = torch.from_numpy(image)
	image = transforms(image).to(config.DEVICE)
	image = image.unsqueeze(0)
	return image


def select_boxes(prediction) -> np.ndarray:
	"""

	:param prediction:
	:type prediction:
	:return: select boxes by threshold on confidence score
	:rtype:
	"""
	boxes = prediction[0]['boxes'].to('cpu').detach().numpy()
	pred_score = list(prediction[0]['scores'].cpu().detach().numpy())
	try:
		score_thresh = int(list(map(lambda i: i < 0.5, pred_score)).index(True))
		boxes = boxes[:score_thresh]
	except ValueError:
		pass
	return boxes


def pair_boxes(gt_boxes, pred_boxes):
	"""
	Appair ground truth and predicted boxes 1-1 using Hungarian algorithm
	:param gt_boxes: array
	:type gt_boxes: bytearray
	:param pred_boxes:
	:type pred_boxes:
	:return: indexes of ground truth and predicted boxes appaired
	:rtype: array, array
	"""
	iou = np_vec_no_jit_iou(np.asarray(gt_boxes), np.asarray(pred_boxes))
	gt_index = [i for i in range(len(gt_boxes))]
	pred_index = [i for i in range(len(pred_boxes))]
	iou = pd.DataFrame(iou, index=list(gt_index), columns=list(pred_index))
	iou_NoUnmatched = iou.loc[iou.sum(axis=1) > 0, iou.sum(axis=0) > 0]
	row_ind, col_ind = linear_sum_assignment(1 - iou_NoUnmatched.values)
	return row_ind, col_ind


def plot_images(pred_boxes, gt_boxes, im, im_bis):
	"""

	:param pred_boxes:
	:type pred_boxes:
	:param gt_boxes:
	:type gt_boxes:
	:param im:
	:type im:
	:param im_bis:
	:type im_bis:
	:return: cv2 plot of stacked images ground truth and predicted images
	:rtype:
	"""
	for [startX, startY, endX, endY] in pred_boxes:
		cv2.rectangle(im_bis, (startX, startY), (endX, endY), (0, 0, 255), 2)
	for box in gt_boxes:
		cv2.rectangle(im, (box[1], box[0]), (box[3], box[2]), (0, 255, 0), 2)
	cv2.putText(im, "ground truth", (20, 20), cv2.FONT_HERSHEY_SIMPLEX,
				0.65, (0, 255, 0), 2)
	cv2.putText(im_bis, "detections", (20, 20), cv2.FONT_HERSHEY_SIMPLEX,
				0.65, (0, 0, 255), 2)
	cv2.imshow("Output", np.hstack([im, im_bis]))
	cv2.waitKey(0)
	cv2.destroyAllWindows()


def predict(args, plot=False):
	"""

	:param args:
	:type args:
	:param plot:
	:type plot:
	:return: prediction loop over test set
	:rtype:
	"""

	test_boxes = json.load(open(config.ANNOTS_TEST_PATH))
	imagePaths = os.listdir(args["input"])

	logging.info("[INFO] loading object detector...")
	model = torch.load(config.MODEL_PATH).to(config.DEVICE)
	model.eval()
	all_tp = []
	all_fn = []
	all_fp = []

	for imagePath in tqdm(imagePaths):
		image = cv2.imread(os.path.join(args['input'], imagePath))
		orig = image.copy()
		orig_bis = image.copy()
		image = preprocess(image)

		prediction = model(image)
		pred_boxes = select_boxes(prediction)

		gt_boxes = test_boxes[imagePath]

		row_ind, col_ind = pair_boxes(gt_boxes, pred_boxes)

		fn = len(gt_boxes)-len(row_ind)
		tp = len(row_ind)
		fp = len(pred_boxes)-len(col_ind)

		all_tp.append(tp)
		all_fn.append(fn)
		all_fp.append(fp)

		(h, w) = orig.shape[:2]
		if plot:
			plot_images(pred_boxes, gt_boxes, orig, orig_bis)

	sum_tp = np.sum(all_tp)
	sum_fn = np.sum(all_fn)
	sum_fp = np.sum(all_fp)
	precision = sum_tp/(sum_fp+sum_tp)
	recall = sum_tp/(sum_fn+sum_tp)
	f1 = 2*recall*precision/(recall+precision)
	logger.info("precision is {}".format(round(precision, 3)))
	logger.info("recall is {}".format(round(recall, 3)))
	logger.info("F1 score is {}".format(round(f1, 3)))


if __name__ == "__main__":
	ap = argparse.ArgumentParser()
	ap.add_argument("-i", "--input", required=True,
					help="path to input image/text file of image paths")
	args = vars(ap.parse_args())

	predict(args)

"""
The following performance was achieved: 
precision is 0.726
recall is 0.581
F1 score is 0.646

"""