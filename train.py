import os

import torch
import data
import models
import options
from test import test

import wandb

def train(opt):
    print(opt)
    with open(os.path.join(opt.checkpoint_dir, 'loss_log.txt'), 'a') as f:
        f.write(str(opt) + '\n')
    dataloader = data.get_dataloader(True, opt.batch, opt.dataset_dir)
    model = models.ResNetModel(opt, train=True)

    total_iter = 0
    loss, wandb_logging_loss = 0.0, 0.0
    while True:
        for batch in dataloader:
            total_iter += 1
            inputs, labels = batch
            outputs = model.optimize_params(inputs, labels).to("cpu")
            _, predicted = torch.max(outputs.data, 1)
            total = labels.size(0)
            correct = (predicted == labels).sum().item()
            batch_acc = 100 * correct / total
            loss += model.get_current_loss()
            wandb_logging_loss += model.get_current_loss()

            if total_iter % opt.wandb_freq == 0:
                wandb.log({
                    "train_error":100 - batch_acc, 
                    "train_loss": wandb_logging_loss / opt.wandb_freq, 
                    "iteration": total_iter
                })
                wandb_logging_loss = 0.0

            if total_iter % opt.print_freq == 0:
                txt = f'iter: {total_iter: 6d}, loss: {loss / opt.print_freq}'
                print(txt)
                with open(os.path.join(opt.checkpoint_dir, 'loss_log.txt'), 'a') as f:
                    f.write(txt + '\n')
                loss = 0.0

            if total_iter % opt.save_params_freq == 0:
                name = f'{total_iter}'
                last_model_path = opt.checkpoint_dir + f'/model_{name}.pth'
                model.save_model(name)
                setattr(opt,'params_path',last_model_path)
                test(opt=opt, iteration=total_iter)

            if total_iter == opt.n_iter:
                name = 'final'
                last_model_path = opt.checkpoint_dir + f'/model_{name}.pth'
                model.save_model('final')
                setattr(opt,'params_path',last_model_path)
                test(opt=opt, iteration=total_iter)
                return

if __name__ == '__main__':
    args = options.parse_args_train()
    train(args)
