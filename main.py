import torch
import time
import torch.nn as nn
import numpy as np
import random
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader

from models.Unet import Model
from data.load_path import set_train_test
from data.dataset import Customdataset
from train.trainer import trainer
from utils.parser import set_parser
from utils.seed import seed_everything
from utils.loss import PerCeptualLoss

def main(args):
    device = args.device if torch.cuda.is_available() else 'cpu'
    print(f"Device : {device}")

    seed_everything(args)

    # Load Data Path
    train_rct, train_ct, train_mr, train_gm, train_wm, train_csf, \
    test_rct, test_ct, test_mr, test_gm, test_wm, test_csf = set_train_test(args)
    print(f"Complete Load data Path")

    # MAKE DATASET
    train_dataset = Customdataset(args,
                                  train_rct,
                                  train_ct,
                                  train_mr,
                                  train_gm,
                                  train_wm,
                                  train_csf)
    train_dataloader = DataLoader(train_dataset, batch_size = args.BATCH_SIZE,
                                  shuffle=args.shuffle)
    
    # Define Model
    model = Model(1, 1).to(device)
    model = nn.DataParallel(model).to(device)

    print(f"{args.model} => {args.region} Segmentation training!")

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.LR, betas=(0.5, 0.99))
    
    # Set loss
    loss_fn1 = nn.BCEWithLogitsLoss().to(device)
    loss_fn2 = nn.L1Loss().to(device)
    loss_fn3 = nn.MSELoss().to(device)

    content_layers, style_layers = args.content_layers, args.style_layers
    loss_fn4 = PerCeptualLoss(device, loss_fn3, content_layers, style_layers)
    loss_hist = trainer(args, train_dataloader, model, optimizer,
                        loss_fn1, loss_fn2, loss_fn3, loss_fn4,
                        device, model_save_path)
    model_save_path = f"./model_parameters/windowed_{args.region}_{args.model}_L1L2BCEPerceptual.pt"
    train_start = time.time()

    train_end = time.time()

    print(f"\nTrain time : {(train_end - train_start)//60} min {(train_end - train_start)%60} sec")
    plt.plot(loss_hist)
    plt.title(f"{args.model}_{args.region} Loss")
    plt.savefig(f"D:\\ACPC\\{args.date}\\windowed_{args.region}_{args.model}v2_L1L2BCEPerceptual.png")

if __name__=='__main__':
    args = set_parser()
    print(args)
    main(args)  
