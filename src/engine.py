import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import PIL
import cv2
from tqdm.notebook import tqdm   # for notebook version
from glob import glob
import ast
import random
import time
from datetime import datetime

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SequentialSampler, RandomSampler
from torch.optim import lr_scheduler

import torchvision


class Learner:
    def __init__(self, model, base_dir, hparams, debug=False):
        self.model = model
        self.base_dir = base_dir
        self.hparams = hparams
        self.debug = debug
        self.epoch = 0
        self.log_path = f'{self.base_dir}/log.txt'
        self.best_summary_loss = 1e5

        param_optimizer = list(self.model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.001},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ] 

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.hparams.lr)
        self.scheduler = self.hparams.scheduler_class(self.optimizer, **self.hparams.scheduler_params)
        self.log('Learner prepared.')

    def fit(self, train_loader, valid_loader):
        for epoch in range(self.hparams.epoch):
            
            lr = self.optimizer.param_groups[0]['lr']
            timestamp = datetime.utcnow().isoformat()
            self.log(f'\n{timestamp}\nLR: {lr}')

            t = time.time()
            summary_loss = self.train(train_loader)
            self.log(f'[RESULT]: Train. Epoch: {self.epoch}, summary_loss: {summary_loss.avg:.5f}, time: {(time.time() - t):.5f}')
            self.save(f'{self.base_dir}/last-checkpoint.bin')

            t = time.time()
            summary_loss = self.validation(valid_loader)

            self.log(f'[RESULT]: Val. Epoch: {self.epoch}, summary_loss: {summary_loss.avg:.5f}, time: {(time.time() - t):.5f}')
            if summary_loss.avg < self.best_summary_loss:
                self.best_summary_loss = summary_loss.avg
                self.model.eval()
                self.save(f'{self.base_dir}/best-checkpoint-{str(self.epoch).zfill(3)}epoch.bin')
                for path in sorted(glob(f'{self.hparams.base_dir}/best-checkpoint-*epoch.bin'))[:-3]:
                    os.remove(path)

            if self.hparams.valid_sched:
                self.scheduler.step(metrics=summary_loss.avg)

            if self.debug:
                print('Debug mode: Done training 1 batch and validation 1 epoch.')
                return 

            self.epoch += 1


    def train(self, train_loader):
        self.model.train()
        summary_loss = AverageMeter()
        t = time.time()
        for step, (images, targets, image_ids) in enumerate(train_loader):
            
            if self.hparams.verbose:
                if step % self.hparams.verbose_step == 0:
                    print(f'\rTraining step {step}/{len(train_loader)}, ' + \
                          f'Summary_loss: {summary_loss.avg:.5f}, ' + \
                          f'Time: {(time.time() - t):.5f}', end='')
            
            images = torch.stack(images).to('cuda').float()
            boxes  = [target['bbox'].to('cuda').float() for target in targets]
            labels = [target['cls'].to('cuda').float() for target in targets]

            loss, _, _ = self.model(images, boxes, labels)
            
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            if self.hparams.step_sched:
                self.scheduler.step()

            batch_size = images.shape[0]
            summary_loss.update(loss.detach().item(), batch_size)

            if self.debug:
                break

        return summary_loss


    def validation(self, val_loader):  # TODO: add IOU
        self.model.eval()
        summary_loss = AverageMeter()
        t = time.time()
        for step, (images, targets, image_ids) in enumerate(val_loader):
            
            if self.hparams.verbose:
                if step % self.hparams.verbose_step == 0:
                    print(f'\rValidation step {step}/{len(val_loader)}, ' + \
                          f'Summary_loss: {summary_loss.avg:.5f}, ' + \
                          f'Time: {(time.time() - t):.5f}', end='')
                    
            with torch.no_grad():
                images = torch.stack(images).to('cuda').float()
                boxes = [target['boxes'].to('cuda').float() for target in targets]
                labels = [target['labels'].to('cuda').float() for target in targets]

                loss, _, _ = self.model(images, boxes, labels)
                batch_size = images.shape[0]
                summary_loss.update(loss.detach().item(), batch_size)

        return summary_loss


    def save(self, path):
        self.model.eval()
        torch.save({
            'model_state_dict': self.model.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_summary_loss': self.best_summary_loss,
            'epoch': self.epoch,
        }, path)


    def load(self, path):
        checkpoint = torch.load(path)
        self.model.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.best_summary_loss = checkpoint['best_summary_loss']
        self.epoch = checkpoint['epoch'] + 1
        

    def log(self, message):
        if self.hparams.verbose:
            print(message)
        with open(self.log_path, 'a+') as logger:
            logger.write(f'{message}\n')
            
            
            
def get_scheduler(hparams: dict, *args):
    
    if hparams.scheduler == 'plateau':
        
        scheduler_class = lr_scheduler.ReduceLROnPlateau
        scheduler_params = dict(
            mode=hparams.sched_monitor,
            factor=hparams.lr_reduce_factor,
            patience=hparams.patience,
            verbose=hparams.sched_verbose, 
            threshold=1e-4,
            threshold_mode='abs',
            cooldown=0, 
            min_lr=1e-8,
            eps=1e-08
        )
        
        
    elif hparams.scheduler == 'one_cycle':
    
        scheduler_class = lr_scheduler.OneCycleLR
        scheduler_params = dict(
            max_lr=hparams.lr, 
            epochs=hparams.epochs, 
            steps_per_epoch=steps_per_epoch, 
            pct_start=hparams.pct_start, 
            anneal_strategy='cos', 
            cycle_momentum=True, 
            div_factor=hparams.div_factor, 
        )

    return scheduler_class, scheduler_params
        
     