import torch
from torch import nn
import os
import glob
import random
import csv
import torch.nn.functional as F
import math, copy
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

'''
class FFNet4(nn.Module):

    def __init__(self):
        super(FFNet4, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(8,16),
            nn.ReLU(),
            nn.Linear(16,32),
            nn.ReLU(),
            nn.Linear(32,2),
            nn.ReLU()
        )


    def forward(self, x):
        """

        :param x: [b 8]
        :return: [b 2] -> [fuel time]
        """
        return self.fc(x)
'''



class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        #print(x.shape)
        #print(x)
        mean = x.mean(-1, keepdim=True)
        #print(mean)
        std = x.std(-1, keepdim=True)
        #print(std)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."
    def __init__(self, d_model, d_ff = 8):
        super(PositionwiseFeedForward, self).__init__()
        # Torch linears have a `b` by default.
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.w_2(F.relu(self.w_1(x)))


class AttentionBlk(nn.Module):

    def __init__(self, embed_dim, num_heads):
        super(AttentionBlk, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.selfattn = nn.MultiheadAttention(embed_dim= self.embed_dim, num_heads= self.num_heads)
        self.norm = LayerNorm(self.embed_dim)
        self.feed_forward = PositionwiseFeedForward(embed_dim)
        self.linear = nn.Linear(self.embed_dim,2)

    def forward(self, x):
        # [1, batch, feature dimension]
        # middle of the window
        q = x[x.shape[0]//2, :, :].unsqueeze(0)
        #print(q.shape)
        x_output, output_weight = self.selfattn(q,x,x)
        x_output = self.norm(q+x_output)
        x_output_ff = self.feed_forward(x_output.squeeze(0))
        #print(x_output_ff.shape)
        x_output = self.norm(x_output.squeeze(0) + x_output_ff)
        #print(x_output.shape())
        return F.relu(self.linear(x_output))

def main():

    net = AttentionBlk(8, 1)
    # [window length, batch size, feature dimension]
    tmp = torch.randn(5, 2, 8)
    # [batch size, output dimension]
    out = net(tmp)
    print(net)
    print("tmp",tmp)
    print("out", out)
    print("fc out:", out.shape)

if __name__ == "__main__":
    main()