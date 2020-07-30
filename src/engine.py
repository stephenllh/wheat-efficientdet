import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import PIL
import cv2
import os
from tqdm.notebook import tqdm   # for notebook version
from glob import glob
import ast
import random
import time
from datetime import datetime
from model import load_model_for_eval
from utils import AverageMeter, Zone
from metrics import calculate_final_score

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SequentialSampler, RandomSampler
from torch.optim import lr_scheduler
from torch.cuda.amp import autocast, GradScaler
import torchvision


class Learner:
    def __init__(self, model, scheduler_class, scheduler_params, hparams):
        self.model = model
        self.hparams = hparams
        self.debug = hparams.debug
        self.root_dir = hparams.root_dir
        self.save_dir = f'{hparams.save_dir}/run_{datetime.now(Zone(+8, False, "GMT")).strftime(f"%Y-%m-%d_%H%M")}'
        self.log_dir = self.save_dir
        self.epoch = 0
        self.best_valid_loss = 1e5
        self.accumulation_steps = hparams.eff_bs // hparams.bs

        param_optimizer = list(self.model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': hparams.wd},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ] 

        self.optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=self.hparams.lr)
        self.scheduler = scheduler_class(self.optimizer, **scheduler_params)
        
        self.log(f'{self.hparams}\n', cancel_print=True)

    def fit(self, train_loader, valid_loader):
        
        if self.hparams.continue_train:
            try:
                self.load(self.hparams.load_path, weights_only=self.hparams.weights_only)
            except:
                checkpoint = torch.load(self.hparams.load_path)
                self.model.model.load_state_dict(checkpoint)
 
        self.scaler = GradScaler()
            
        for epoch in range(self.hparams.epoch):
             
            lr = self.optimizer.param_groups[0]['lr']
            self.log(f'\n{datetime.now(Zone(+8, False, "GMT")).strftime(f"%d-%m-%Y %H:%M")}')
            self.log(f'\nEpoch {epoch+1}/{self.hparams.epoch}')
            self.log(f'\nInitial learning rate for epoch {epoch}: {lr:.4e}')
            
            # Training loop
            t = time.time()
            train_loss = self.train(train_loader)
            tt = time.time() - t
            self.log(f'\n[RESULT]: Training loss: {train_loss.avg:.5f}, Time taken: {tt//60:.0f}m {tt%60:.0f}s')
            self.save('last-cp.bin')
            
            # Validation loop
            t = time.time()
            valid_loss, iou = self.validation(valid_loader)
            tt = time.time() - t
            self.log(f'\n[RESULT]: Validation loss: {valid_loss.avg:.5f}, IOU: {iou:.5f}, Time taken: {tt//60:.0f}m {tt%60:.0f}s')
            
            if valid_loss.avg < self.best_valid_loss:
                self.best_valid_loss = valid_loss.avg
                self.model.eval()
                self.save(f'ckpt-e{str(epoch).zfill(3)}.bin')
                for path in sorted(glob(f'{self.hparams.root_dir}/ckpt-e*.bin'))[:-3]:
                    os.remove(path)

            if self.hparams.valid_sched:
                self.scheduler.step(metrics=valid_loss.avg)

            if self.debug:
                print('Debug mode: Done training 1 batch and validating 1 epoch.')
                return 

            self.epoch += 1


    def train(self, train_loader):
            
        self.model.train()
        
        train_loss = AverageMeter()
        t = time.time()
        for step, (images, targets, image_ids) in enumerate(train_loader):
            if self.hparams.verbose:
                if (step + 1) % self.hparams.verbose_step == 0:
                    lr = self.optimizer.param_groups[0]['lr']
                    print(
                        f'\rTraining step {step+1}/{len(train_loader)}, ' + \
                        f'Learning rate {lr:.4e}, ' + \
                        f'Training loss: {train_loss.avg:.5f}, ' + \
                        f'Time taken: {(time.time() - t):.1f}s', end=''
                    )
            
            images = torch.stack(images).to('cuda').float()
            boxes  = [target['boxes'].to('cuda').float() for target in targets]
            labels = [target['labels'].to('cuda').float() for target in targets]
            
            if self.hparams.fp16:
                with autocast():
                    loss, _, _ = self.model(images, boxes, labels)
                    loss /= self.accumulation_steps
                self.scaler.scale(loss).backward()
            else:
                loss, _, _ = self.model(images, boxes, labels)
                loss.backward()
                
            if (step + 1) % self.accumulation_steps == 0:
                if self.hparams.fp16:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    # self.scaler.unscale_(self.optimizer)    
                else:
                    self.optimizer.step()
                    
                self.optimizer.zero_grad() 
                        
                if self.hparams.step_sched:
                    self.scheduler.step()

            batch_size = images.shape[0]
            train_loss.update(loss.detach().item() * self.accumulation_steps, batch_size)

            if self.debug:
                self.save('last-cp.bin')
                break 

        return train_loss


    def validation(self, val_loader):  # TODO: add IOU
        
        self.model.eval()
        valid_loss = AverageMeter()
        t = time.time()
        for step, (images, targets, image_ids) in enumerate(val_loader):
            
            if self.hparams.verbose:
                if step % self.hparams.verbose_step == 0:
                    print(f'\rValidation step {step+1}/{len(val_loader)}, ' + \
                          f'Validation loss: {valid_loss.avg:.5f}, ' + \
                          f'Time taken: {(time.time() - t):.1f}s', end='')
                 
            all_predictions = []   
            with torch.no_grad():
                images = torch.stack(images).to('cuda').float()
                boxes = [target['boxes'].to('cuda').float() for target in targets]
                labels = [target['labels'].to('cuda').float() for target in targets]

                loss, _, _ = self.model(images, boxes, labels)
                batch_size = images.shape[0]
                valid_loss.update(loss.detach().item(), batch_size)
                
                # Calculate IOU
                eval_model = load_model_for_eval(os.path.join(self.save_dir, 'last-cp.bin'), variant=self.hparams.model_variant)
                preds = eval_model(images, torch.tensor([1]*images.shape[0]).float().cuda())
                targets['boxes'][:, [0,1,2,3]] = targets['boxes'][:, [1,0,3,2]]   # revert back to xyxy
                
                for i in range(images.shape[0]):
                    boxes = preds[i].detach().cpu().numpy()[:, :4]    
                    scores = preds[i].detach().cpu().numpy()[:,4]
                    boxes[:, 2] = boxes[:, 2] + boxes[:, 0]
                    boxes[:, 3] = boxes[:, 3] + boxes[:, 1]
                    all_predictions.append({
                        'pred_boxes': (boxes*2).clip(min=0, max=1023).astype(int),
                        'scores': scores,
                        'gt_boxes': (targets[i]['boxes'].cpu().numpy()*2).clip(min=0, max=1023).astype(int),
                        'image_id': image_ids[i],
                    })
                
                best_final_score, best_score_threshold = 0, 0
                for score_threshold in np.arange(0, 1, 0.01):
                    final_score = calculate_final_score(all_predictions, score_threshold)
                    if final_score > best_final_score:
                        best_final_score = final_score
                        best_score_threshold = score_threshold

        return valid_loss, best_final_score


    def save(self, name):
        
        self.model.eval()
        torch.save({
            'model_state_dict': self.model.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_valid_loss': self.best_valid_loss,
            'epoch': self.epoch,
        }, os.path.join(self.save_dir, name))
      

    def load(self, path, weights_only):

        checkpoint = torch.load(path)
        self.model.model.load_state_dict(checkpoint['model_state_dict'])
        
        if weights_only:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            self.best_valid_loss = checkpoint['best_valid_loss']
            self.epoch = checkpoint['epoch'] + 1
    

    def log(self, message, cancel_print=False):
        
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        
        if self.hparams.verbose and not cancel_print:
            print(message)
            
        with open(f'{self.log_dir}/log.txt', 'a+') as logger:
            logger.write(f'{message}\n')
            
               
def get_scheduler(hparams: dict):
    
    if hparams.scheduler == 'plateau':
        scheduler_class = lr_scheduler.ReduceLROnPlateau
        scheduler_params = dict(
            mode=hparams.valid_sched_metric,
            factor=hparams.lr_reduce_factor,
            patience=hparams.patience,
            verbose=hparams.sched_verbose, 
            threshold=1e-4,
            threshold_mode='abs',
            cooldown=0, 
            min_lr=1e-8,
            eps=1e-08
        )
        hparams.valid_sched = True
        hparams.step_sched = False
        
    elif hparams.scheduler == 'one_cycle':
        scheduler_class = lr_scheduler.OneCycleLR
        scheduler_params = dict(
            max_lr=hparams.lr, 
            epochs=hparams.epoch, 
            steps_per_epoch=hparams.steps_per_epoch,
            pct_start=hparams.pct_start, 
            anneal_strategy='cos', 
            cycle_momentum=True, 
            div_factor=hparams.div_factor, 
        )
        hparams.step_sched = True
        hparams.valid_sched = False

    return scheduler_class, scheduler_params
        
     