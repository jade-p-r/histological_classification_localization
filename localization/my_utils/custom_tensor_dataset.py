# import the necessary packages
from torch.utils.data import Dataset
from PIL import Image
import os
import torch
import json


class CustomTensorDataset(Dataset):
	def __init__(self, root, image_list, boxes, transforms=None):
		self.root = root
		self.imgs = image_list
		self.transforms = transforms
		self.boxes = json.load(open(boxes))

	def __getitem__(self, idx):
		img_path = os.path.join(self.root, self.imgs[idx])
		img = Image.open(img_path).convert("RGB")

		im_boxes = self.boxes[self.imgs[idx]]
		num_objs = len(im_boxes)
		boxes = []
		for box in im_boxes:
			xmin = box[1]
			xmax = box[3]
			ymin = box[0]
			ymax = box[2]
			boxes.append([xmin, ymin, xmax, ymax])

		boxes = torch.as_tensor(boxes, dtype=torch.float32)
		labels = torch.ones((num_objs,), dtype=torch.int64)

		image_id = torch.tensor([idx])
		if boxes.numel() != 0:
			area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
		else:
			area = 0
		iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

		target = {}
		target["boxes"] = boxes
		target["labels"] = labels
		target["image_id"] = image_id
		target["area"] = area
		target["iscrowd"] = iscrowd

		if self.transforms:
			img, target = self.transforms(img, target)

		return img, target

	def __len__(self):
		return len(self.imgs)
