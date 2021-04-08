
import visdom
import torch
from torch import nn, optim
import torchvision
import os, glob
import random, csv
from torch.utils.data import Dataset, DataLoader
from obddata import ObdData
from nets import AttentionBlk
import numpy as np
import time
import torch.nn.functional as F

batchsz = 32
lr = 1e-3
epochs = 3000
#epochs = 0
torch.manual_seed(1234)
output_dimension = 2
feature_dimension = 8
head_number = 1
train_path_length = 10
test_path_length = 10
device = torch.device("cuda")
#device = torch.device("cpu")

# in Google Colab
ckpt_path = "/content/drive/MyDrive/Colab_Notebooks/DeepLearning/best.mdl"
data_root = "/content/drive/MyDrive/Colab_Notebooks/DeepLearning/model_data"
# local
#ckpt_path = "best.mdl"
#data_root = "model_data"

train_db = ObdData(root=data_root,mode = "train",percentage=20, path_length=train_path_length)
val_db = ObdData(root=data_root,mode="val",percentage=20, path_length=train_path_length)
test_db = ObdData(root=data_root,mode="test",percentage=20, path_length=test_path_length)
train_loader = DataLoader(train_db, batch_size=batchsz, num_workers=2)
val_loader = DataLoader(val_db, batch_size=batchsz, num_workers=2)
test_loader = DataLoader(test_db, batch_size=batchsz, num_workers=2)

'''
viz = visdom.Visdom()
startup_sec = 1
while not viz.check_connection() and startup_sec > 0:
    time.sleep(0.1)
    startup_sec -= 0.1

assert viz.check_connection(), 'No connection could be formed quickly'
'''

def mape_loss(pred, label):
    '''

    :param pred: [batchsz, output dimension]
    :param label:
    :return:
    '''
    #print(pred.shape)
    p = pred[:, 0]
    l = label[:, 0]
    mask = l != 0
    loss_energy = torch.mean(torch.abs((p[mask] - l[mask]) / l[mask]))
    if loss_energy == float("inf"):
        print(label)
    #loss_time = torch.mean(torch.abs((pred[:,1]-label[:,1])/(label[:,1]+1e-6)))
    return loss_energy

def test(model, loader):
    """

    :param preds:
    :param labels:
    :return: mape of energy
    """
    loss_mape = 0
    loss_mse = 0
    cnt = 0
    for x, y in loader:
        x,y = x.to(device), y.to(device)
        with torch.no_grad():
            label_segment = torch.zeros(y.shape[0], output_dimension).to(device)
            pred_segment = torch.zeros(y.shape[0], output_dimension).to(device)
            for i in range(x.shape[1]):
                # [batch size, window length, feature dimension]
                x_segment = x[:, i, :, :]
                # [window size, batch size,feature dimension]
                x_segment = x_segment.transpose(0, 1).contiguous()
                # [batch size, output dimension]
                pred = model(x_segment)
                # print("pred", pred)
                pred_segment += pred
                # print("pred_segment", pred_segment)
                # print(result_segment.shape)
                # [window size, batch size, output dimension]
                y_segment = y[:, i, :, :].transpose(0, 1).contiguous()
                # print("y_segment", y_segment.shape)
                label = y_segment[y_segment.shape[0] // 2, :, :]
                # print("label", label)
                label_segment += label
                # print(label_segment.shape)
                # break
            loss_mse += F.mse_loss(label_segment, pred_segment)
            loss_mape += mape_loss(label_segment, pred_segment)

            cnt += 1
    return loss_mape / cnt , loss_mse/cnt


'''
def evalute(model, loader):

    loss = 0
    cnt = 0
    for x,y in loader:
        # x,y = x.to(device), y.to(device)
        with torch.no_grad():
            label_segment = torch.zeros(y.shape[0], output_dimension)
            pred_segment = torch.zeros(y.shape[0], output_dimension)
            for i in range(x.shape[1]):
                # [batch size, window length, feature dimension]
                x_segment = x[:, i, :, :]
                # [window size, batch size,feature dimension]
                x_segment = x_segment.transpose(0, 1).contiguous()
                # [batch size, output dimension]
                pred = model(x_segment)
                # print("pred", pred)
                pred_segment += pred
                # print("pred_segment", pred_segment)
                # print(result_segment.shape)
                # [window size, batch size, output dimension]
                y_segment = y[:, i, :, :].transpose(0, 1).contiguous()
                # print("y_segment", y_segment.shape)
                label = y_segment[y_segment.shape[0] // 2, :, :]
                # print("label", label)
                label_segment += label
                # print(label_segment.shape)
                # break
            loss += F.mse_loss(label_segment, pred_segment)
            cnt += 1
    return loss/cnt
'''


def main():
    model = AttentionBlk(feature_dimension, head_number)

    if os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path))
        print("load from ckpt!")
    else:
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    model.to(device)
    print(model)

    # this code is very important! It initialises the parameters with a
    # range of values that stops the signal fading or getting too big.
    # See this blog for a mathematical explanation.
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-9)
    # optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    # optimizer = optim.Adam(model.parameters(),lr=lr)
    p = sum(map(lambda p: p.numel(), model.parameters()))
    print("number of parameters:", p)

    criteon = nn.MSELoss()

    '''
    a_pred = torch.from_numpy(np.array([[1,3],[1,2],[1,2]],dtype=float))
    a_label = torch.from_numpy(np.array([[0,4],[1,2],[1,2]], dtype=float))
    print(a_pred.shape)
    test = torch.zeros(3,2)
    print(test)
    test+= a_pred
    loss = criteon(a_pred,a_label)
    print(test)

    '''

    best_mse, best_epoch = float("inf"), 0
    global_step = 0

    for epoch in range(epochs):

        for step, (x, y) in enumerate(train_loader):
            # print(step)
            x, y = x.to(device), y.to(device)
            # [batch size, path length, window length, feature dimension/output dimension]
            # print(x.shape, y.shape)
            loss_total = 0

            label_segment = torch.zeros(y.shape[0], output_dimension).to(device)
            pred_segment = torch.zeros(y.shape[0], output_dimension).to(device)

            for i in range(x.shape[1]):
                # [batch size, window length, feature dimension]
                x_segment = x[:, i, :, :]
                # [window size, batch size,feature dimension]
                x_segment = x_segment.transpose(0, 1).contiguous()
                # [batch size, output dimension]
                pred = model(x_segment)
                # print("pred",pred)
                # print(pred_segment.shape, pred.shape)
                pred_segment += pred
                # print("pred_segment",pred_segment)
                # print(result_segment.shape)
                # [window size, batch size, output dimension]
                y_segment = y[:, i, :, :].transpose(0, 1).contiguous()
                # print("y_segment", y_segment.shape)
                label = y_segment[y_segment.shape[0] // 2, :, :]
                # print("label", label)
                label_segment += label
                # print(label_segment.shape)
                # break
            loss = criteon(label_segment, pred_segment)
            # print(label_segment,pred_segment)
            # print(loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # break
        # break
        # viz.line([loss], [global_step], win="mape",update="append")
        global_step += 1
        if epoch % 10 == 0:
            val_mape, val_mse = test(model, val_loader)
            print("epoch", epoch, "val_mape", val_mape, "val_mse", val_mse)
            if val_mse < best_mse:
                best_epoch = epoch
                best_mse = val_mse
                torch.save(model.state_dict(), ckpt_path)
            # viz.line([val_mape], [global_step], win="val_mape", update="append")
    print("best_mse", best_mse, "best_epoch", best_epoch)

    model.load_state_dict(torch.load(ckpt_path))
    print("load from ckpt!")
    test_mape, test_mse = test(model, test_loader)
    # test_mape_tt, test_mape_e, test_mape_t = evalute(model, test_loader)
    print("test_mape", test_mape)

    print("test_mse", test_mse)

'''

def main_test():
    num_epochs = 6000
    class Linear(nn.Module):
        def __init__(self):
            super(Linear, self).__init__()
            self.linear = nn.Linear(4, 4)
        def forward(self,x):
            return self.linear(x)

    x = torch.from_numpy(np.array([[1, 3,3,4], [1, 2,3,2], [1, 2,5,3]], dtype=float)).to(torch.float32)
    print(x.shape)
    label = 1000*x
    print(label)
    model = Linear()
    criteon = nn.MSELoss()
    learning_rate = 0.001
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    for i in range(num_epochs):


        y = model(x)
        loss = criteon(y,label)
        if i % 10 == 0:
            print("y", y)
            print("loss",loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def main_test_online():
    import matplotlib.pyplot as plt

    # Hyper-parameters 定义迭代次数， 学习率以及模型形状的超参数
    input_size = 1
    output_size = 1
    num_epochs = 60
    learning_rate = 0.001

    # Toy dataset  1. 准备数据集
    x_train = np.array([[3.3], [4.4], [5.5], [6.71], [6.93], [4.168],
                        [9.779], [6.182], [7.59], [2.167], [7.042],
                        [10.791], [5.313], [7.997], [3.1]], dtype=np.float32)

    y_train = np.array([[1.7], [2.76], [2.09], [3.19], [1.694], [1.573],
                        [3.366], [2.596], [2.53], [1.221], [2.827],
                        [3.465], [1.65], [2.904], [1.3]], dtype=np.float32)

    # Linear regression model  2. 定义网络结构 y=w*x+b 其中w的size [1,1], b的size[1,]
    model = nn.Linear(input_size, output_size)

    # Loss and optimizer 3.定义损失函数， 使用的是最小平方误差函数
    criterion = nn.MSELoss()
    # 4.定义迭代优化算法， 使用的是随机梯度下降算法
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    loss_dict = []
    # Train the model 5. 迭代训练
    for epoch in range(num_epochs):
        # Convert numpy arrays to torch tensors  5.1 准备tensor的训练数据和标签
        inputs = torch.from_numpy(x_train)
        targets = torch.from_numpy(y_train)

        # Forward pass  5.2 前向传播计算网络结构的输出结果
        outputs = model(inputs)
        # 5.3 计算损失函数
        loss = criterion(outputs, targets)

        # Backward and optimize 5.4 反向传播更新参数
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 可选 5.5 打印训练信息和保存loss
        loss_dict.append(loss.item())
        if (epoch + 1) % 5 == 0:
            print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, loss.item()))

    # Plot the graph 画出原y与x的曲线与网络结构拟合后的曲线
    predicted = model(torch.from_numpy(x_train)).detach().numpy()
    plt.plot(x_train, y_train, 'ro', label='Original data')
    plt.plot(x_train, predicted, label='Fitted line')
    plt.legend()
    plt.show()

    # 画loss在迭代过程中的变化情况
    plt.plot(loss_dict, label='loss for every epoch')
    plt.legend()
    plt.show()

'''



if __name__ == '__main__':
    main()

