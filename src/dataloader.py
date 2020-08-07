import torch
import cv2
import random
import numpy as np
from utils import collate_fn
import albumentations
from albumentations.pytorch.transforms import ToTensorV2  # must import manually because 'pytorch' is not imported in __init__
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SequentialSampler, RandomSampler


class Dataset:
    
    def __init__(self, data_path, df, image_ids, transforms=None, do_cutmix=True, test=False):
        self.data_path = data_path
        self.df = df
        self.image_ids = image_ids
        self.transforms = transforms
        self.do_cutmix = False if test else do_cutmix

    def __getitem__(self, index):
        image_id = self.image_ids[index]
        
        if self.do_cutmix and random.random() > 0.5:
            image, boxes = self.load_cutmix_image_and_boxes(index)
        else:
            image, boxes = self.load_image_and_boxes(index)
        
        labels = torch.ones((boxes.shape[0],), dtype=torch.int64)   # there is only one class
        
        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': torch.tensor([index])
        }

        if self.transforms:
            while True:
                sample = self.transforms(**{'image': image, 'bboxes': target['boxes'], 'labels': labels})
                if len(sample['bboxes']) > 0:
                    image = sample['image']
                    # target['boxes'] = torch.stack(tuple(map(torch.tensor, zip(*sample['bboxes'])))).permute(1, 0)
                    target['boxes'] = torch.tensor(sample['bboxes'])
                    target['boxes'][:, [0,1,2,3]] = target['boxes'][:, [1,0,3,2]]  # y,x,y,x for efficientdet_pytorch repo CocoDataset class (self.yxyx=True for PyTorch)
                    target['labels'] = torch.stack(sample['labels'])
                    break

        return image, target, image_id


    def __len__(self):
        return self.image_ids.shape[0]


    def load_image_and_boxes(self, index):

        # process images into RGB and rescale to 0-1
        image_id = self.image_ids[index]
        image = cv2.imread(f'{self.data_path}/train/{image_id}.jpg', cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image /= 255.0

        # process all the bounding boxes (labels) for this image
        df_images = self.df[self.df['image_id'] == image_id]   # get all the rows for the image_id
        boxes = df_images[['x', 'y', 'w', 'h']].values
        boxes[:, 2] = boxes[:, 0] + boxes[:, 2]   # change width  to x_max
        boxes[:, 3] = boxes[:, 1] + boxes[:, 3]   # change height to y_max

        return image, boxes

    def load_cutmix_image_and_boxes(self, index, imsize=1024):
        '''
        This implementation of cutmix author:  https://www.kaggle.com/nvnnghia 
        Refactoring and adaptation: https://www.kaggle.com/shonenkov
        '''
        
        w, h = imsize, imsize
        s = imsize // 2

        # randomly pick any coordinate (xc, yc) from the center quarter square in the image    
        xc, yc = [int(random.uniform(imsize * 0.25, imsize * 0.75)) for _ in range(2)]  # center x, y
        
        # Generate multiple random indexes for cutmix: (data_index, image_corner_index: 0 for top left, etc.)
        indexes = [index] + [random.randint(0, self.image_ids.shape[0] - 1) for _ in range(3)]

        # Initializing: image: filled with 3D array of ones, boxes: empty list
        result_image = np.full((w, h, 3), 1, dtype=np.float32)
        result_boxes = []

        # Below code ensures the bounding boxes are not cut off and do not exceed image borders when performing cutmix
        for i, index in enumerate(indexes):
            image, boxes = self.load_image_and_boxes(index)
            
            if i == 0:   # top left
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc    # x_min, y_min, x_max, y_max (large image)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h    # x_min, y_min, x_max, y_max (small image)
            
            elif i == 1:  # top right
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            
            elif i == 2:  # bottom left
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, max(xc, w), min(y2a - y1a, h)
            
            elif i == 3:  # bottom right
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)
            
            result_image[y1a : y2a, x1a : x2a] = image[y1b : y2b, x1b : x2b]
            padw = x1a - x1b
            padh = y1a - y1b

            boxes[:, 0] += padw
            boxes[:, 1] += padh
            boxes[:, 2] += padw
            boxes[:, 3] += padh

            result_boxes.append(boxes)

        result_boxes = np.concatenate(result_boxes, 0)
        np.clip(result_boxes[:, 0:], 
                a_min=0, 
                a_max=2 * s, 
                out=result_boxes[:, 0:])
        result_boxes = result_boxes.astype(np.int32)
        result_boxes = result_boxes[np.where((result_boxes[:,2]-result_boxes[:,0]) * (result_boxes[:,3]-result_boxes[:,1]) > 0)]

        return result_image, result_boxes
    
    

def get_train_loader(data_path, df, image_ids, transforms, do_cutmix, batch_size, num_workers):
    train_dataset = Dataset(data_path, df, image_ids, transforms=transforms, do_cutmix=do_cutmix, test=False)
    dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=RandomSampler(train_dataset),
        pin_memory=False,
        drop_last=True,
        num_workers=num_workers,
        collate_fn=collate_fn
    )
    
    return dataloader


def get_valid_loader(data_path, df, image_ids, transforms, batch_size, num_workers):
    valid_dataset = Dataset(data_path, df, image_ids, transforms=transforms, test=True)
    dataloader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=SequentialSampler(valid_dataset),
        pin_memory=False,
        num_workers=num_workers,
        collate_fn=collate_fn
    )
    
    return dataloader