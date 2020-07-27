import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import random
import os


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