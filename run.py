import options
from train import train
import wandb

if __name__ == '__main__':

    # Parse train args
    args = options.parse_args_train()

    # Wandb config
    config = {
    "depth": args.n,
    "batch": args.batch,
    "wandb_freq": args.wandb_freq,
    "learning rate": args.lr,
    "momentum": args.momentum,
    "weight decay": args.weight_decay,
    "decay_lr_1": args.decay_lr_1,
    "decay_lr_2": args.decay_lr_2,
    "lr_decay_rate": args.lr_decay_rate,
    "n_iter": args.n_iter,
    "optimizer": args.optimizer,
    }

    if args.run_name:
        wandb.init(entity="nishantaswani", project="advml_h2p4", config=config, mode=args.wandb, name=args.run_name)
    else:
        wandb.init(entity="nishantaswani", project="advml_h2p4", config=config, mode=args.wandb)

    # Train
    train(args)    
