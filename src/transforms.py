import albumentations
from albumentations.pytorch.transforms import (
    ToTensorV2,
)  # must import manually because 'pytorch' is not imported in __init__


def get_train_augs(hparams: dict):

    aug_list = [
        albumentations.RandomSizedCrop(
            min_max_height=(800, 800), height=1024, width=1024, p=hparams.crop
        ),
        albumentations.OneOf(
            [
                albumentations.HueSaturationValue(
                    hue_shift_limit=0.2,
                    sat_shift_limit=0.2,
                    val_shift_limit=0.2,
                    p=hparams.hue,
                ),
                albumentations.RandomBrightnessContrast(
                    brightness_limit=0.2, contrast_limit=0.2, p=hparams.bright_contrast
                ),
            ],
            p=0.9,
        ),
        albumentations.ToGray(p=hparams.gray),
        albumentations.HorizontalFlip(p=hparams.hflip),
        albumentations.VerticalFlip(p=hparams.vflip),
        albumentations.Resize(height=hparams.img_size, width=hparams.img_size, p=1),
        albumentations.Cutout(
            num_holes=hparams.cut_holes,
            max_h_size=64,
            max_w_size=64,
            fill_value=0,
            p=hparams.cutout,
        ),
        ToTensorV2(p=1.0),
    ]

    bbox_params = dict(
        format="pascal_voc", min_area=0, min_visibility=0, label_fields=["labels"]
    )

    return albumentations.Compose(
        aug_list, p=1.0, bbox_params=albumentations.BboxParams(**bbox_params)
    )


def get_valid_augs(hparams: dict):

    aug_list = [
        albumentations.Resize(height=hparams.img_size, width=hparams.img_size, p=1.0),
        ToTensorV2(p=1.0),
    ]
    bbox_params = dict(
        format="pascal_voc", min_area=0, min_visibility=0, label_fields=["labels"]
    )

    return albumentations.Compose(
        aug_list, p=1.0, bbox_params=albumentations.BboxParams(**bbox_params)
    )
