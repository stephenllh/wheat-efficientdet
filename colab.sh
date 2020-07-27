pip install -U -q albumentations
pip install -q timm
pip install -q omegaconf
pip install -q --upgrade --force-reinstall --no-deps kaggle
git clone https://github.com/stephenllh/wheat_efficientdet.git

python -c '
import os; os.environ["KAGGLE_CONFIG_DIR"] = "drive/My Drive/.kaggle"
'

unzip -q drive/My Drive/DL/kaggle/global_wheat_detection/data.zip -d wheat_efficientdet/input
mkdir wheat_efficientdet/pretrained_models
unzip -q efficientdet.zip -d wheat_efficientdet/pretrained_models

cd wheat_efficientdet/src

python train.py \
--epoch=100 \
--model-variant=d0 \
--bs=8 \