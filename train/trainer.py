import torch
import time

from utils.seed import seed_everything

def trainer(args, train_dataloader, model, optimizer, 
                    loss_fn1, loss_fn2, loss_fn3, loss_fn4,
                    device, model_path):
    seed_everything(args)

    model.train()

    size = len(train_dataloader)

    train_loss_hist = []

    for epoch in range(args.EPOCHS):
        epoch_start = time.time()  
        print(f"Start epoch : {epoch+1}/{args.EPOCHS}")
        train_loss_item=0 

        for batch, (train_ct, train_mr) in enumerate(train_dataloader): 
            train_ct = train_ct.to(device).float()
            train_mr = train_mr.to(device).float()

            train_output = model(train_ct)
    
            train_loss1 = loss_fn1(train_output, train_mr) # BCE
            train_loss2 = loss_fn2(train_output, train_mr) # L1
            train_loss3 = loss_fn3(train_output, train_mr) # L2
            train_loss4_1, train_loss4_2  = loss_fn4.cal_perceptual_loss(train_output, train_mr)
            train_loss4 = (train_loss4_1 + train_loss4_2) # Perceptual

            train_loss = train_loss1 + train_loss2 + train_loss3 + train_loss4

            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
            if batch % 30 == 0:
                print(f"Batch Loss : {train_loss.item():>.5f} BCE : {train_loss1.item():>.5f} L1 : {train_loss2.item():>.5f} L2 : {train_loss3.item():>.5f} Per : {train_loss4.item():.5f} {batch}/{size}")

            train_loss_item += train_loss.item()

        train_loss_hist.append(train_loss_item/size)

        print(f"Loss : {train_loss_item/size:.5f}")

        epoch_end = time.time()
        print(f"End epoch : {epoch+1}/{args.EPOCHS}")
        print(f"Epoch time : {(epoch_end-epoch_start)//60} min {(epoch_end-epoch_start)%60} sec")
        print()

        if epoch % 20 == 0 :
            torch.save(model.state_dict(), f"./model_parameters/windowed_{args.region}_{args.model}v2_L1L2BCEPerceptual.pt")

    torch.save(model.state_dict(), model_path)

    return train_loss_hist