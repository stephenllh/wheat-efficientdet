import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold


def process_data(df):
    df = pd.read_csv('train.csv')
    bboxes = np.stack(df['bbox'].apply(lambda x: np.fromstring(x[1: -1], sep=',')))

    for i, column in enumerate(['x', 'y', 'w', 'h']):
        df[column] = bboxes[:, i]

    df = df.drop(columns=['bbox'])
    return df
    
    
def create_folds(df):
    df_folds = df[['image_id']].copy()

    # Group the dataframe by image_id (because 1 image_id can appear in multiple rows) and get the bbox_count
    df_folds.loc[:, 'bbox_count'] = 1   # each row corresponds to 1 bbox
    df_folds = df_folds.groupby('image_id').count()

    # Match the source to each image_id
    df_folds['source'] = df[['image_id', 'source']].groupby('image_id').first()['source']  # max() achieves the same

    # Create stratify group 
    df_folds['stratify_group'] = np.char.add(
        df_folds['source'].values.astype(str),
        df_folds['bbox_count'].apply(lambda x: f'_{x//15}').values.astype(str)   # 15 is rather arbritrary
    )  

    # Initialize fold as -1
    df_folds['fold'] = -1

    # Assign fold indices
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for fold_idx, (train_idx, valid_idx) in enumerate(skf.split(X=df_folds.index, y=df_folds['stratify_group'])):
        df_folds.loc[df_folds.iloc[valid_idx].index, 'fold'] = fold_idx