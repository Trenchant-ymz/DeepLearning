import torch
from torch import nn
import torch.nn.functional as F
from node2vec import N2V

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
    def __init__(self, d_model, d_ff = 64):
        super(PositionwiseFeedForward, self).__init__()
        # Torch linears have a `b` by default.
        self.net_dropped = torch.nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, 1)
        )

    def forward(self, x):
        return self.net_dropped(x)


class ConstGat(nn.Module):

    def __init__(self, n2v_dim, attention_dim, feature_dim, embedding_dim, num_heads, output_dimension):
        super(ConstGat, self).__init__()
        self.n2v = N2V('node2vec.mdl')
        self.linearContextual = nn.Linear(n2v_dim, attention_dim)
        self.embedding_dim = embedding_dim
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.background_dim = embedding_dim[1] + embedding_dim[2] + 1
        self.linearQ = nn.Linear(self.background_dim+n2v_dim, attention_dim)
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
        self.selfattn = nn.MultiheadAttention(embed_dim= attention_dim, num_heads= self.num_heads)
        self.traffic_dim = embedding_dim[0] + sum(embedding_dim[3:]) + feature_dim- 1
        self.linearTraffic = nn.Linear(self.traffic_dim, attention_dim)
        self.norm = LayerNorm(self.total_embed_dim)
        self.feed_forward = PositionwiseFeedForward(2*attention_dim+self.background_dim)
        self.activate = nn.ReLU()


    def forward(self, x, c, segment):
        # x -> [ batch, window size, feature dimension]
        # c -> [batch, number of categorical features, window size]

        # [batch_sz, window_sz, embedding dim]
        '''
        embedded_road_type = self.embedding_road_type(c[:,0,:])
        embedded_time_stage = self.embedding_time_stage(c[:, 1, :])
        embedded_week_day = self.embedding_week_day(c[:, 2, :])
        embedded_lanes = self.embedding_lanes (c[:, 3, :])
        embedded_bridge = self.embedding_bridge(c[:, 4, :])
        embedded_endpoint_u = self.embedding_endpoint_u(c[:, 5, :])
        embedded_endpoint_v = self.embedding_endpoint_v(c[:, 6, :])
        '''
        # [batch_sz, window_sz, embedding dim]

        # representation + convolution
        segmentEmbed = self.n2v.embed(segment[:,0])
        segmentEmbed = torch.cat([segmentEmbed, self.n2v.embed(segment[:,1])], dim=-1)
        segmentEmbed = torch.cat([segmentEmbed, self.n2v.embed(segment[:, 2])], dim=-1)
        contextual = F.relu(self.linearContextual(segmentEmbed))

        # background information time; day; mass
        background = self.embedding_time_stage(c[:, 1, :])
        background = torch.cat([background, self.embedding_week_day(c[:, 2, :]), x[:,:,1].unsqueeze(-1)], dim=-1)

        catConBack = torch.cat([contextual, background], dim=-1)
        catConBack = catConBack.transpose(0, 1).contiguous()
        q = F.relu(self.linearQ(catConBack))
        # [ window size, batch,  2* dense]

        embedded = self.embedding_road_type(c[:,0,:])
        embedded = torch.cat([embedded, self.embedding_lanes(c[:, 3, :])], dim=-1)
        embedded = torch.cat([embedded, self.embedding_bridge(c[:, 4, :])], dim=-1)
        embedded = torch.cat([embedded, self.embedding_endpoint_u(c[:, 5, :])], dim=-1)
        embedded_6 = self.embedding_endpoint_v(c[:, 6, :])
        embedded = torch.cat([embedded, embedded_6], dim=-1)
        # [ batch, window size, feature dimension+ sum embedding dimension]
        trafficPred = torch.cat([x[:,:,0].unsqueeze(-1), x[:,:,2:], embedded], dim=-1)


        # [ window size, batch,  feature dimension+ sum embedding dimension]
        trafficPred = trafficPred.transpose(0, 1).contiguous()

        kv = F.relu(self.linearTraffic(trafficPred))

        # x -> [windowsz, batch, feature dimension+ sum embedding dimension]
        x_output, output_weight = self.selfattn(q,kv,kv)

        x_output = torch.cat([catConBack,x_output], dim=-1)
        x_output = self.feed_forward(x_output)
        return self.activate(x_output)

def testNet():
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
    testNet()