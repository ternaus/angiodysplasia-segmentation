"""
Script generates predictions, splitting original images into tiles, and assembling prediction back together
"""
import argparse
from prepare_train_val import get_split
from dataset import AngyodysplasiaDataset
import cv2
from models import UNet, UNet11, UNet16, AlbuNet34
import torch
from pathlib import Path
from tqdm import tqdm
import numpy as np
import utils
# import prepare_data
from torch.utils.data import DataLoader
from torch.nn import functional as F

from transforms import (ImageOnly,
                        Normalize,
                        CenterCrop,
                        DualCompose)

img_transform = DualCompose([
    CenterCrop(512),
    ImageOnly(Normalize())
])


def get_model(model_path, model_type):
    """

    :param model_path:
    :param model_type: 'UNet', 'UNet11', 'UNet16', 'AlbuNet34'
    :return:
    """

    num_classes = 1

    if model_type == 'UNet11':
        model = UNet11(num_classes=num_classes)
    elif model_type == 'UNet16':
        model = UNet16(num_classes=num_classes)
    elif model_type == 'AlbuNet34':
        model = AlbuNet34(num_classes=num_classes)
    elif model_type == 'UNet':
        model = UNet(num_classes=num_classes)
    else:
        model = UNet(num_classes=num_classes)

    state = torch.load(str(model_path))
    state = {key.replace('module.', ''): value for key, value in state['model'].items()}
    model.load_state_dict(state)

    if torch.cuda.is_available():
        return model.cuda()

    model.eval()

    return model


def predict(model, from_file_names, batch_size: int, to_path):
    loader = DataLoader(
        dataset=AngyodysplasiaDataset(from_file_names, transform=img_transform, mode='predict'),
        shuffle=False,
        batch_size=batch_size,
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available()
    )

    for batch_num, (inputs, paths) in enumerate(tqdm(loader, desc='Predict')):
        inputs = utils.variable(inputs, volatile=True)

        outputs = model(inputs)

        for i, image_name in enumerate(paths):
            mask = (F.sigmoid(outputs[i, 0]).data.cpu().numpy() * 255).astype(np.uint8)

            h, w = mask.shape

            full_mask = np.zeros((576, 576))
            full_mask[32:32 + h, 32:32 + w] = mask

            (to_path / args.model_type).mkdir(exist_ok=True, parents=True)

            cv2.imwrite(str(to_path / args.model_type / (Path(paths[i]).stem + '.png')), full_mask)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--model_path', type=str, default='data/models/UNet', help='path to model folder')
    arg('--model_type', type=str, default='UNet', help='network architecture',
        choices=['UNet', 'UNet11', 'UNet16', 'AlbuNet34'])
    arg('--batch-size', type=int, default=4)
    arg('--fold', type=int, default=-1, choices=[0, 1, 2, 3, 4, -1], help='-1: all folds')
    arg('--workers', type=int, default=12)

    args = parser.parse_args()

    if args.fold == -1:
        for fold in [0, 1, 2, 3, 4]:
            _, file_names = get_split(fold)

            print(len(file_names))

            model = get_model(str(Path(args.model_path).joinpath('model_{fold}.pt'.format(fold=fold))),
                              model_type=args.model_type)

            print('num file_names = {}'.format(len(file_names)))

            output_path = Path(args.model_path)
            output_path.mkdir(exist_ok=True, parents=True)

            predict(model, file_names, args.batch_size, output_path)
    else:
        _, file_names = get_split(args.fold)
        model = get_model(str(Path(args.model_path).joinpath('model_{fold}.pt'.format(fold=args.fold))),
                          model_type=args.model_type)

        print('num file_names = {}'.format(len(file_names)))

        output_path = Path(args.model_path)
        output_path.mkdir(exist_ok=True, parents=True)

        predict(model, file_names, args.batch_size, output_path)
