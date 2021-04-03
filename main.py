
import visdom
import torch
from torch import nn, optim
import torchvision
import os, glob
import random, csv
from torch.utils.data import Dataset, DataLoader
from obddata import ObdData
from nets import FFNet4
import numpy as np
import time

batchsz = 32
lr = 1e-4
epochs = 10
torch.manual_seed(1234)
# device = torch.device("cuda")

train_db = ObdData("model_data",mode = "train",percentage=20)
val_db = ObdData("model_data",mode="val",percentage=20)
test_db = ObdData("model_data",mode="test",percentage=20)
train_loader = DataLoader(train_db, batch_size=batchsz, num_workers=4)
val_loader = DataLoader(val_db, batch_size=batchsz, num_workers=4)
test_loader = DataLoader(test_db, batch_size=batchsz, num_workers=4)

'''
viz = visdom.Visdom()
startup_sec = 1
while not viz.check_connection() and startup_sec > 0:
    time.sleep(0.1)
    startup_sec -= 0.1

assert viz.check_connection(), 'No connection could be formed quickly'
'''



def mape(preds, labels):
    """

    :param preds:
    :param labels:
    :return: total loss, loss of energy, loss of time
    """
    loss_energy = torch.mean(torch.abs((preds[:, 0] - labels[:, 0]) / (labels[:, 0]+1e-6)))
    loss_time = torch.mean(torch.abs((preds[:,1]-labels[:,1])/(labels[:,1]+1e-6)))
    return loss_time+loss_energy, loss_energy, loss_time


def evalute(model, loader):
    loss_tt, loss_e, loss_t, cnt = 0, 0, 0, 0
    for x,y in loader:
        # x,y = x.to(device), y.to(device)
        with torch.no_grad():
            result = model(x)
            l_tt, l_e, l_t = mape(result, y)
            loss_tt += l_tt
            loss_e += l_e
            loss_t += l_t
            cnt += 1
    return loss_tt/cnt, loss_e/cnt, loss_t/cnt


def main():

    model = FFNet4()
    # model.to(device)
    print(model)
    optimizer = optim.Adam(model.parameters(),lr=lr)
    p = sum(map(lambda p: p.numel(), model.parameters()))
    tensor_a = torch.tensor([[0, 3],[5,6], [1, 2],[3,4]])
    tensor_b = torch.tensor([[1,2],[3,4],[1,2],[3,4]])
    #print(tensor_a[:,1])
    # m_a, m_b, m_c = mape(tensor_a, tensor_b)
    # print(m_a, m_b, m_c)
    criteon = nn.MSELoss()
    best_mape, best_epoch = float("inf"), 0
    global_step = 0
    # viz.line([0],[-1], win="mape", opts=dict(title="mape"))
    # viz.line([0], [-1], win="val_mape", opts=dict(title="val_mape"))
    for epoch in range(epochs):

        for step, (x,y) in enumerate(train_loader):
            # x,y = x.to(device), y.to(device)

            result = model(x)
            loss, loss_e, loss_t = mape(result,y)
            #print(result, y)
            print(loss, loss_e, loss_t)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            #break
        break
        # viz.line([loss], [global_step], win="mape",update="append")
        global_step+= 1
        if epoch%2 == 0:
            val_mape, _, _ = evalute(model, val_loader)
            print(val_mape)
            if val_mape < best_mape:
                best_epoch = epoch
                best_mape = val_mape
                torch.save(model.state_dict(),"best.mdl")
            # viz.line([val_mape], [global_step], win="val_mape", update="append")
    print("best_mape",best_mape,"best_epoch", best_epoch)

    model.load_state_dict(torch.load("best.mdl"))
    print("load from ckpt!")

    test_mape_tt, test_mape_e, test_mape_t = evalute(model, test_loader)
    print(test_mape_tt, test_mape_e, test_mape_t)


if __name__ == '__main__':
    main()

