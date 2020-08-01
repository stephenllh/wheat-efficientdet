import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import random
import os
from datetime import datetime, tzinfo, timedelta


class AverageMeter:
    def __init__(self): 
        self.reset()

    def reset(self): 
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        
        
def view_dataset(dataset, idx):
    image, target, image_id = dataset[idx]
    boxes = target['bbox'].cpu().numpy().astype(np.int16)
    image = image.permute(1, 2, 0).cpu().numpy()

    fig, ax = plt.subplots(1, 1, figsize=(16, 8))

    for box in boxes:
        cv2.rectangle(image, 
                      pt1=(box[1], box[0]), 
                      pt2=(box[3], box[2]), 
                      color=(0, 1, 0), thickness=2)
        
    ax.set_axis_off()
    ax.imshow(image)
    
    
def collate_fn(batch):
    return tuple(zip(*batch))


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    

def show_result(sample_id, preds, gt_boxes):
    sample = cv2.imread(f'{TRAIN_ROOT_PATH}/{sample_id}.jpg', cv2.IMREAD_COLOR)
    sample = cv2.cvtColor(sample, cv2.COLOR_BGR2RGB)

    fig, ax = plt.subplots(1, 1, figsize=(16, 8))

    for pred_box in preds:
        cv2.rectangle(
            sample,
            (pred_box[0], pred_box[1]),
            (pred_box[2], pred_box[3]),
            (220, 0, 0), 2
        )

    for gt_box in gt_boxes:    
        cv2.rectangle(
            sample,
            (gt_box[0], gt_box[1]),
            (gt_box[2], gt_box[3]),
            (0, 0, 220), 2
        )

    ax.set_axis_off()
    ax.imshow(sample)
    ax.set_title("RED: Predicted | BLUE - Ground-truth")
    

class Zone(tzinfo):
    def __init__(self, offset, isdst, name):
        self.offset = offset
        self.isdst = isdst
        self.name = name

    def utcoffset(self, dt):
        return timedelta(hours=self.offset) + self.dst(dt)

    def dst(self, dt):
        return timedelta(hours=1) if self.isdst else timedelta(0)

    def tzname(self,dt):
         return self.name
     
     
def nullable_string(value):
    if not value:
        return None
    return value