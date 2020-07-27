import os
from datetime import datetime
import pandas as pd
import argparse
from data import process_data, create_folds
from dataloader import get_train_loader, get_valid_loader
from transforms import get_train_augs, get_valid_augs
from model import get_model
from engine import get_scheduler, Learner
from utils import seed_everything


parser = argparse.ArgumentParser(description='Wheat detection with EfficientDet')

# Directories
parser.add_argument('--root-dir', default='../', type=str, help='directory of data')
parser.add_argument('--data-dir', default='../input', type=str, help='directory of data')
parser.add_argument('--model-dir', default='../pretrained_models', type=str, help='directory of downloaded efficientnet models')
parser.add_argument('--save-dir', default='../', type=str, help='directory of saved models')

# Training fold
parser.add_argument('--subset', default=1.0, type=float, help='subset of data')
parser.add_argument('--fold', default=0, type=int, help='fold number')

# Augmentations
parser.add_argument('--crop', default=0.5, type=float, help='proba of random sized crop')
parser.add_argument('--hue', default=0.9, type=float, help='proba of hue saturation')
parser.add_argument('--bright-contrast', default=0.9, type=float, help='proba of RandomBrightnessContrast')
parser.add_argument('--gray', default=0.01, type=float, help='proba of converting to grayscale')
parser.add_argument('--hflip', default=0.5, type=float, help='proba of horizontal flip')
parser.add_argument('--vflip', default=0.5, type=float, help='proba of vertical flip')
parser.add_argument('--img-size', default=512, type=int, help='proba of horizontal flip')
parser.add_argument('--cut-holes', default=8, type=int, help='number of holes in cutout')
parser.add_argument('--cutout', default=0.5, type=float, help='proba of cutout')
parser.add_argument('--cutmix', default=True, type=bool, help='do cutmix in Dataset or not')

# Model variant selection
parser.add_argument('--model-variant', default='d5', type=str, help='model variant: d0 to d7')

# Training
parser.add_argument('--epoch', '-e', type=int, required=True, help='number of epochs')
parser.add_argument('--lr', default=2e-4, type=float, help='learning rate')
parser.add_argument('--bs', default=4, type=int, help='batch size')
parser.add_argument('--num-workers', default=4, type=int, help='num workers')

# Scheduler
parser.add_argument('--scheduler', default='plateau', type=str, help='scheduler class: choose from ["plateau", "one_cycle"]')
parser.add_argument('--step-sched', default=False, type=bool, help='use step scheduler or not')
parser.add_argument('--valid-sched', default=True, type=bool, help='use valid scheduler or not')
parser.add_argument('--sched-verbose', default=False, type=bool, help='verbosity in the scheduler')
parser.add_argument('--verbose', default=True, type=bool, help='verbosity in the Learner')
parser.add_argument('--verbose-step', default=1, type=int, help='verbosity step in the Learner')
parser.add_argument('--debug', default=False, type=bool, help='debug mode or not in Learner')

# Scheduler args: ReduceLROnPlateau
parser.add_argument('--valid-sched-metric', default='min', type=str, help='the mode argument in valid scheduler')
parser.add_argument('--lr-reduce-factor', default=0.5, type=float, help='reduce factor in ReduceLROnPlateau')
parser.add_argument('--patience', default=2, type=int, help='patience only for valid scheduler')

# Scheduler args: OneCycle
parser.add_argument('--pct_start', default=0.3, type=float, help='pct of total no. iterations to start annealing')
parser.add_argument('--div_factor', default=10, type=int, help='lr reducion factor at the beginning')

# Seed
parser.add_argument('--seed', default=42, type=int, help='seed')

# Save model
parser.add_argument('--saved_model_name', default='model', type=str, help='name of saved model after training')


args = parser.parse_args()

#print(args)


    
def run():
    seed_everything(args.seed)
    
    df = pd.read_csv(os.path.join(args.data_dir, 'train.csv'))
    df = process_data(df, args.subset)
    df_folds = create_folds(df)
    
    train_image_ids = df_folds[df_folds['fold'] != args.fold].index.values
    valid_image_ids = df_folds[df_folds['fold'] == args.fold].index.values
    
    train_loader = get_train_loader(args.data_dir, df, train_image_ids, transforms=get_train_augs(args), do_cutmix=args.cutmix, 
                                    batch_size=args.bs, num_workers=args.num_workers)
    
    valid_loader = get_valid_loader(args.data_dir, df, valid_image_ids, transforms=get_valid_augs(args), 
                                    batch_size=args.bs, num_workers=args.num_workers)
    
    model = get_model(args.model_variant, model_dir=args.model_dir).cuda()
    
    # Get scheduler
    scheduler_class, scheduler_params = get_scheduler(args)
    args.scheduler_class = scheduler_class
    args.scheduler_params = scheduler_params
    
    if args.scheduler == 'one_cycle':
        steps_per_epoch = len(train_image_ids) // args.bs
        scheduler_class, scheduler_params = get_scheduler(args, steps_per_epoch)
        
    else: 
        scheduler_class, scheduler_params = get_scheduler(args)
        
    args.scheduler_class = scheduler_class
    args.scheduler_params = scheduler_params
    
    learner = Learner(model, root_dir=args.root_dir, hparams=args, debug=args.debug)
    learner.fit(train_loader, valid_loader)
    
    learner.save(f'../models/{args.saved_model_name}.pth')
    print(f'Model is saved with name of {args.saved_model_name}.pth')
    


if __name__ == '__main__':
    run()