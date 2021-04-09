import torch
from torch import nn
import torch.nn.functional as F


class LayerNorm(nn.Module):
    "Construct a layernorm module."
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
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
        # q -> [1, batch, feature dimension]
        # middle of the window
        q = x[x.shape[0]//2, :, :].unsqueeze(0)
        # x -> [windowsz, batch, feature dimension]
        x_output, output_weight = self.selfattn(q,x,x)
        x_output = self.norm(q+x_output)
        x_output_ff = self.feed_forward(x_output.squeeze(0))
        x_output = self.norm(x_output.squeeze(0) + x_output_ff)
        return F.relu(self.linear(x_output))

def main():
    # test nets
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