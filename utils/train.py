"""Import libraries"""
import numpy as np
from easydict import EasyDict

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from sklearn.metrics import *




def train(train_loader, valid_loader, test_loader, model, args):
    opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer=opt, eta_min=args.lr * 0.001, T_max=50
    )

    criterion = nn.CrossEntropyLoss()

    for epoch in range(args.EPOCHS):
        train_mode = "Train"
        train_loop(
            train_mode=train_mode,
            data_loader=train_loader,
            model=model,
            criterion=criterion,
            opt=opt,
            lr_scheduler=lr_scheduler,
            epochs=epoch,
        )

        train_mode = "Valid"
        train_loop(
            train_mode=train_mode,
            data_loader=valid_loader,
            model=model,
            criterion=criterion,
            epochs=epoch,
        )

    train_mode = "Test"
    test_loop(
        train_mode=train_mode, data_loader=test_loader, model=model
    )


def train_loop(
    train_mode,
    data_loader,
    model,
    criterion=None,
    opt=None,
    lr_scheduler=None,
    epochs=None,
):
    sum_loss = 0
    pred_list = []
    true_list = []

    if train_mode == "Train":
        model.train()
    elif train_mode == "Valid" and train_mode == "Test":
        model.eval()

    for idx, (x, y) in tqdm(enumerate(data_loader)):
        if train_mode == "Train":
            opt.zero_grad()

        x = x.unsqueeze(1).to(device="cuda")
        y = y.flatten().to(device="cuda")
        pred = model(x)
        loss = criterion(pred, y)

        if train_mode == "Train":
            loss.backward()
            opt.step()

        sum_loss += loss.item()
        confidence = torch.softmax(pred, dim=-1)
        prediction = torch.argmax(confidence, dim=-1)
        pred_list.append(prediction.cpu().numpy())
        true_list.append(y.cpu().numpy())

    if lr_scheduler != None and train_mode == "Train":
        lr_scheduler.step()

    pred_list = np.concatenate(pred_list)
    true_list = np.concatenate(true_list)

    acc = accuracy_score(true_list, pred_list)
    sum_loss /= idx + 1


    if train_mode != "Test":
        print(
            f"[Epochs {epochs + 1}, {train_mode} Results] Acc.: {acc:.4f}, Loss: {sum_loss:.4f}"
        )

    else:
        print(f"[{train_mode} Results] Acc.: {acc:.4f}")

    if train_mode == "Valid":
        confusion = confusion_matrix(true_list, pred_list)
        print(confusion)

def test_loop(
    train_mode,
    data_loader,
    model,
):
    pred_list = []
    model.eval()

    for _, (x) in tqdm(enumerate(data_loader)):
       

        x = x.unsqueeze(1).to(device="cuda")
        pred = model(x)

        confidence = torch.softmax(pred, dim=-1)
        prediction = torch.argmax(confidence, dim=-1)
        pred_list.append(prediction.cpu().numpy())

    pred_list = np.concatenate(pred_list)



    