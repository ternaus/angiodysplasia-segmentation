from pathlib import Path
import argparse
import cv2
import numpy as np
from tqdm import tqdm


def general_dice(y_true, y_pred):
    if y_true.sum() == 0:
        if y_pred.sum() == 0:
            return 1
        else:
            return 0

    return dice(y_true == 1, y_pred == 1)


def general_jaccard(y_true, y_pred):
    if y_true.sum() == 0:
        if y_pred.sum() == 0:
            return 1
        else:
            return 0

    return jaccard(y_true == 1, y_pred == 1)


def jaccard(y_true, y_pred):
    intersection = (y_true * y_pred).sum()
    union = y_true.sum() + y_pred.sum() - intersection
    return (intersection + 1e-15) / (union + 1e-15)


def dice(y_true, y_pred):
    return (2 * (y_true * y_pred).sum() + 1e-15) / (y_true.sum() + y_pred.sum() + 1e-15)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    arg = parser.add_argument

    arg('--train_path', type=str, default='data/train/angyodysplasia/masks', help='path where train images with ground truth are located')
    arg('--target_path', type=str, default='predictions/UNet', help='path with predictions')
    args = parser.parse_args()

    result_dice = []
    result_jaccard = []

    for file_name in tqdm(list(Path(args.train_path).glob('*'))):
        y_true = (cv2.imread(str(file_name), 0) > 255 * 0.5).astype(np.uint8)

        pred_file_name = Path(args.target_path) / (file_name.stem.replace('_a', '') + '.png')

        y_pred = (cv2.imread(str(pred_file_name), 0) > 255 * 0.5).astype(np.uint8)

        result_dice += [dice(y_true, y_pred)]
        result_jaccard += [jaccard(y_true, y_pred)]

    print('Dice = ', np.mean(result_dice), np.std(result_dice))
    print('Jaccard = ', np.mean(result_jaccard), np.std(result_jaccard))
