import argparse
import torch

import data
import models

def train(opt):
    print(opt)
    dataloader = data.get_dataloader(True, opt.batch, opt.dataset_dir)
    model = models.ResNetModel(opt)

    total_iter = 0
    loss = 0.0

    while True:
        for batch in dataloader:
            total_iter += 1
            inputs, labels = batch
            model.optimize_params(inputs, labels)
            loss += model.get_current_loss()

            if total_iter % 10 == 0:
                print(f'iter: {total_iter: 6d}, loss: {loss / 10}')
                loss = 0.0

            if total_iter % 1000 == 0:
                model.save_model(f'{total_iter // 1000}k')
            
            if total_iter == 64000:
                model.save_model(f'{total_iter // 1000}k')
                break


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', help='network complexity', type=int, default=3)
    parser.add_argument('--batch', help='batch size', type=int, default=128)
    parser.add_argument('--checkpoint_dir', default='./checkpoint')
    parser.add_argument('--dataset_dir', default='./dataset')
    parser.add_argument('--is_train', action='store_true')
    args = parser.parse_args()
    
    return args


if __name__ == '__main__':
    args = parse_args()
    train(args)
