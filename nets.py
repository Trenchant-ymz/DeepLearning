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
    def __init__(self, d_model, d_ff = 32):
        super(PositionwiseFeedForward, self).__init__()
        # Torch linears have a `b` by default.
        self.net_dropped = torch.nn.Sequential(
            nn.Linear(d_model, d_ff),
            #nn.Dropout(0.5),
            #nn.ReLU(),
            #nn.Linear(d_ff, d_ff),
            #nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
        #self.w_1 = nn.Linear(d_model, d_ff)

        #self.w_2 = nn.Linear(d_ff, d_ff)
        #self.w_3 = nn.Linear(d_ff, d_model)
    def forward(self, x):
        #return self.w_3(F.relu(self.w_2(F.relu(self.w_1(x)))))
        return self.net_dropped(x)


class AttentionBlk(nn.Module):

    def __init__(self, feature_dim, embedding_dim, num_heads, output_dimension):
        super(AttentionBlk, self).__init__()
        self.embedding_dim = embedding_dim
        self.feature_dim = feature_dim
        self.total_embed_dim =  self.feature_dim + sum(self.embedding_dim)
        self.output_dimension = output_dimension
        self.num_heads = num_heads
        # embedding layers for 7 categorical features
        # "road_type", "time_stage", "week_day", "lanes", "bridge", "endpoint_u", "endpoint_v", "trip_id"
        # 0 represents Unknown
        # 0-21
        self.embedding_road_type = nn.Embedding(22, self.embedding_dim[0])
        # 0-6
        self.embedding_time_stage = nn.Embedding(7, self.embedding_dim[1])
        # 0-7
        self.embedding_week_day = nn.Embedding(8, self.embedding_dim[2])
        # 0-8
        self.embedding_lanes = nn.Embedding(9, self.embedding_dim[3])
        # 0-1
        self.embedding_bridge = nn.Embedding(2, self.embedding_dim[4])
        # 0-16
        self.embedding_endpoint_u = nn.Embedding(17, self.embedding_dim[5])
        self.embedding_endpoint_v = nn.Embedding(17, self.embedding_dim[6])
        self.selfattn = nn.MultiheadAttention(embed_dim= self.total_embed_dim, num_heads= self.num_heads)
        self.norm = LayerNorm(self.total_embed_dim )
        self.feed_forward = PositionwiseFeedForward(self.total_embed_dim)
        self.linear = nn.Linear(self.total_embed_dim,self.output_dimension)


    def forward(self, x, c):
        # x -> [ batch, window size, feature dimension]
        # c -> [batch, number of categorical features, window size]

        # [batch_sz, window_sz, embedding dim]
        '''
        embedded_road_type = self.embedding_road_type(c[:,0,:])
        embedded_time_stage = self.embedding_time_stage(c[:, 1, :])
        embedded_week_day = self.embedding_week_day(c[:, 2, :])
        embedded_lanes = self.embedding_lanes(c[:, 3, :])
        embedded_bridge = self.embedding_bridge(c[:, 4, :])
        embedded_endpoint_u = self.embedding_endpoint_u(c[:, 5, :])
        embedded_endpoint_v = self.embedding_endpoint_v(c[:, 6, :])
        '''
        # [batch_sz, window_sz, embedding dim]
        embedded = self.embedding_road_type(c[:,0,:])
        embedded = torch.cat([embedded,self.embedding_time_stage(c[:, 1, :])], dim=-1)
        embedded = torch.cat([embedded, self.embedding_week_day(c[:, 2, :])], dim=-1)
        embedded = torch.cat([embedded, self.embedding_lanes(c[:, 3, :])], dim=-1)
        embedded = torch.cat([embedded, self.embedding_bridge(c[:, 4, :])], dim=-1)
        embedded = torch.cat([embedded, self.embedding_endpoint_u(c[:, 5, :])], dim=-1)
        embedded = torch.cat([embedded, self.embedding_endpoint_v(c[:, 6, :])], dim=-1)
        #print(x.shape)
        # [ batch, window size, feature dimension+ sum embedding dimension]
        x = torch.cat([x, embedded], dim=-1)
        # [ window size, batch,  feature dimension+ sum embedding dimension]
        x = x.transpose(0, 1).contiguous()
        #print(x.shape)
        # q -> [1, batch, feature dimension+ sum embedding dimension]
        # middle of the window
        q = x[x.shape[0] // 2, :, :].unsqueeze(0)
        # x -> [windowsz, batch, feature dimension+ sum embedding dimension]
        x_output, output_weight = self.selfattn(q,x,x)
        # x_output -> [1, batchsz, feature dimension+ sum embedding dimension]
        x_output = self.norm(q+x_output)
        #print("x_output.shape", x_output.shape)
        x_output_ff = self.feed_forward(x_output.squeeze(0))
        #print("x_output_ff.shape", x_output_ff.shape)
        x_output = self.norm(x_output.squeeze(0) + x_output_ff)
        return F.relu(self.linear(x_output))

def main():
    # test nets
    net = AttentionBlk(feature_dim=6,embedding_dim=[4,2,2,2,2,4,4],num_heads=1,output_dimension=1)
    # [batch size, window length,  feature dimension]
    tmp = torch.randn(1, 3, 6)
    # [batch, categorical_dim, window size]
    c = torch.randint(1, (1, 7, 3))
    print(tmp.shape,c.shape)
    # [batch size, output dimension]
    out = net(tmp,c)

    print(net)
    print("tmp",tmp)
    print("out", out)
    print("fc out:", out.shape)

if __name__ == "__main__":
    main()