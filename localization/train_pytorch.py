import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torch
from my_utils.custom_tensor_dataset import CustomTensorDataset
import torch.utils.data
import os
from engine import train_one_epoch, evaluate
import utils
#from torchvision import transforms as T
import transforms as T
import json
from my_utils import config
import random

def create_model():
    """
    Creates Pytorch model from pretrained model Faster RCNN - Resent 50
    :return: torch pretrained model
    :rtype: torch model
    """
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True, box_detections_per_img=10000)
    num_classes = 2  # 1 class (nuclei) + background
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
        #transforms.append(T.Normalize(mean=config.MEAN, std=config.STD))

    return T.Compose(transforms)


def main():
    images = os.listdir("train")
    train_boxes = json.load(open("boxes_train.json"))
    train_boxes_nonempty = {k: v for k, v in train_boxes.items() if v}
    train_images_list = [image for image in images if image in train_boxes_nonempty.keys()]
    random.shuffle(train_images_list)
    val_images_list = train_images_list[:int(0.1 * len(train_images_list))]
    train_images_list = [image for image in train_images_list if not image in val_images_list]

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    dataset = CustomTensorDataset('train', train_images_list, 'boxes_train.json', get_transform(train=True))
    dataset_val = CustomTensorDataset('train', val_images_list, 'boxes_train.json', get_transform(train=False))

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=True, num_workers=os.cpu_count(),
        collate_fn=utils.collate_fn)

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, batch_size=1, shuffle=False, num_workers=os.cpu_count(),
        collate_fn=utils.collate_fn)

    model = create_model()
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005,
                                momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.1)

    num_epochs = 5

    for epoch in range(num_epochs):
        print("doing epoch {}".format(epoch))
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
        lr_scheduler.step()
        evaluate(model, data_loader_val, device=device)
        torch.save(model, "output/pyt_detector_3.pth")
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
"""
the model achieved the built in performance on the validation set : 
Averaged stats: model_time: 0.4636 (0.4647)  evaluator_time: 0.1532 (0.1749)
Accumulating evaluation results...
DONE (t=0.03s).
IoU metric: bbox
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.487
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.794
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.547
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.488
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.168
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = -1.000
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.013
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.122
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.574
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.576
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.200
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = -1.000
"""