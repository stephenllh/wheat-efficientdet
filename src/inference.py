import os
import pandas as pd
import argparse
from data import process_data, create_folds
from dataloader import get_train_loader, get_valid_loader
from transforms import get_train_augs, get_valid_augs
from model import get_model
from engine import get_scheduler, Learner
from utils import seed_everything


parser = argparse.ArgumentParser(description="Wheat detection with EfficientDet")

# Directories
parser.add_argument("--root-dir", default="../", type=str, help="directory of data")
parser.add_argument(
    "--data-dir", default="../input", type=str, help="directory of data"
)
parser.add_argument(
    "--model-dir",
    default="../pretrained_models",
    type=str,
    help="directory of downloaded efficientnet models",
)
parser.add_argument(
    "--save-dir", default="../models", type=str, help="directory of saved models"
)
parser.add_argument(
    "--load-dir", default="../models", type=str, help="directory of saved models"
)

# Training fold
parser.add_argument("--subset", default=1.0, type=float, help="subset of data")
parser.add_argument("--fold", default=0, type=int, help="fold number")

# Model variant selection
parser.add_argument(
    "--model-variant", "-m", required=True, type=str, help="model variant: d0 to d7"
)

# Training
parser.add_argument("--num-workers", default=4, type=int, help="num workers")
parser.add_argument("--fp16", default=True, type=bool, help="use fp16 or not")

# Seed
parser.add_argument("--seed", default=42, type=int, help="seed")

# Save model
parser.add_argument(
    "--save-name", default="model", type=str, help="name of saved model after training"
)

# Load model
parser.add_argument(
    "--load-path", default="model", type=str, help="dir + name of loaded model"
)
parser.add_argument(
    "--weights-only",
    default=True,
    type=bool,
    help="True: use as transfer learning. False: continue from checkpoint.",
)
parser.add_argument(
    "--continue-train",
    default=False,
    type=bool,
    help="Continue from saved model or not",
)

args = parser.parse_args()


def run():
    seed_everything(args.seed)

    df = pd.read_csv(os.path.join(args.data_dir, "train.csv"))
    df = process_data(df, args.subset)
    df_folds = create_folds(df)

    train_image_ids = df_folds[df_folds["fold"] != args.fold].index.values
    valid_image_ids = df_folds[df_folds["fold"] == args.fold].index.values

    train_loader = get_train_loader(
        args.data_dir,
        df,
        train_image_ids,
        transforms=get_train_augs(args),
        do_cutmix=args.cutmix,
        batch_size=args.bs,
        num_workers=args.num_workers,
    )

    valid_loader = get_valid_loader(
        args.data_dir,
        df,
        valid_image_ids,
        transforms=get_valid_augs(args),
        batch_size=args.bs,
        num_workers=args.num_workers,
    )

    model = get_model(args.model_variant, model_dir=args.model_dir).cuda()

    if args.scheduler == "one_cycle":
        args.steps_per_epoch = len(train_image_ids) // args.bs
        scheduler_class, scheduler_params = get_scheduler(args)

    else:
        scheduler_class, scheduler_params = get_scheduler(args)

    learner = Learner(model, scheduler_class, scheduler_params, hparams=args)
    learner.fit(train_loader, valid_loader)


if __name__ == "__main__":
    run()
