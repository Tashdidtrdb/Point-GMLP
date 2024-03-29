from argparse import ArgumentParser
import torch
from torch import nn, optim
from models.point_gmlp import PointGMLP
from dataloader import SHRECLoader
from train import train, test
import math
import time
import matplotlib.pyplot as plt
import os
import wandb
import yaml
from utils.loss import LabelSmoothingLoss
from utils.scheduler import WarmUpLR, get_scheduler
from utils.misc import*
from utils.opt import*
from utils.config_parser import get_config
################################
# run for the model
################################

def plot_chart(stats):
    plt.plot(stats[0], 'r', label = 'train_acc')
    plt.plot(stats[2], 'g', label = 'val_acc')
    plt.title('accuracies')
    plt.legend(loc = "lower right")

    plt.show()
    plt.plot(stats[3], 'b', label = 'val_loss')
    plt.plot(stats[1], 'y', label = 'train_loss')
    plt.title('losses')
    plt.legend(loc = "upper right")
    plt.show()

def training_pipeline(config):

    #create the model
    model = get_model(config["hparams"]["model"])
    model = model.to(config["hparams"]["device"])
    print("Successfully created point glmp model")
    
    #Get the number of parameters of the model
    num_params = count_params(model)
    print(f"PointGMLP model has {num_params} parameters.")

    

    ###########  Optimizer Definition  ###########
    optimizer = get_optimizer(model, config["hparams"])

    ###########  Criterion Definition  ###########
    criterion = LabelSmoothingLoss(
        num_classes = config["hparams"]["model"]["model_kwargs"]["num_classes"],
        smoothing = config["hparams"]["l_smooth"]
    ) 

    train_set = SHRECLoader(framerate = 32)
    val_set   = SHRECLoader(framerate = 32, phase = 'validation')
    test_set  = SHRECLoader(framerate = 32, phase = 'test')
    
    train_loader = torch.utils.data.DataLoader(
        dataset = train_set,
        batch_size = config["hparams"]["batch_size"],
        shuffle = True,
        num_workers = 0,
    )

    val_loader = torch.utils.data.DataLoader(
        dataset = val_set,
        batch_size = config["hparams"]["batch_size"],
        shuffle = True,
        num_workers = 0,
    )

    test_loader = torch.utils.data.DataLoader(
        dataset = test_set,
        shuffle = True,
        num_workers = 0,
    )

    # lr scheduler
    schedulers = {
        "warmup": None,
        "scheduler": None
    }
    if config["hparams"]["scheduler"]["n_warmup"]:
        schedulers["warmup"] = WarmUpLR(optimizer, total_iters=len(train_loader) * config["hparams"]["scheduler"]["n_warmup"])

    if config["hparams"]["scheduler"]["scheduler_type"] is not None:
        total_iters = len(train_loader) * max(1, (config["hparams"]["n_epochs"] - config["hparams"]["scheduler"]["n_warmup"]))
        schedulers["scheduler"] = get_scheduler(optimizer, config["hparams"]["scheduler"]["scheduler_type"], total_iters)

    if config["exp"]["wandb"]:
        if config["exp"]["wandb_api_key"] is not None:
            with open(config["exp"]["wandb_api_key"], "r") as f:
                os.environ["WANDB_API_KEY"] = f.read()
   
        wandb.init(project=config["exp"]["proj_name"], name=config["exp"]["exp_name"], config=config["hparams"], entity = config["exp"]["entity"])
    
    stats_train = train(model, train_loader, val_loader, criterion, optimizer, config["hparams"]["n_epochs"], config["hparams"]["device"], schedulers, config)
    test_acc, test_loss = test(stats_train[4], criterion, test_loader, config["hparams"]["device"])

    plot_chart(stats_train)
    print("\nTest Accuracy :",test_acc,"\nTest Loss : ", test_loss)
    print("Completed run.")

def main(args):
    config = get_config(args.conf)

    seed_everything(config["hparams"]["seed"])
    training_pipeline(config)


if __name__ == "__main__":
    parser = ArgumentParser("Driver code.")
    parser.add_argument("--conf", type=str, required=True, help="Path to config.yaml file.")
    args = parser.parse_args()

    main(args)

    