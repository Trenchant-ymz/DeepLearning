import torch
from torch import nn
import torch.nn.functional as F
from node2vec import N2V
import pickle
from torch_geometric.nn import Node2Vec

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
    def __init__(self, d_model, d_out, d_ff = 64):
        super(PositionwiseFeedForward, self).__init__()
        # Torch linears have a `b` by default.
        self.layer1 = nn.Linear(d_model, d_ff)
        self.act1 = nn.ReLU()
        self.layer2 = nn.Linear(d_ff, d_out)


    def forward(self, x):
        x = self.layer1(x)
        x = self.act1(x)
        return self.layer2(x)


class ConstGat(nn.Module):

    def __init__(self, n2v_dim, attention_dim, feature_dim, embedding_dim, num_heads, output_dimension, windowsz = 3):
        super(ConstGat, self).__init__()
        open_file = open("edge_index.pkl", "rb")
        edge_index = pickle.load(open_file)
        open_file.close()
        self.n2v = Node2Vec(edge_index, embedding_dim=32, walk_length=20,
                            context_size=10, walks_per_node=10,
                            num_negative_samples=1, p=1, q=1, sparse=True)

        #self.n2v = N2V('node2vec.mdl')
        self.linearContextual = nn.Linear(n2v_dim, attention_dim)
        self.embedding_dim = embedding_dim
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.output_dimension = output_dimension
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
        self.selfattn = nn.MultiheadAttention(embed_dim= attention_dim, num_heads= self.num_heads, batch_first=True)
        self.traffic_dim = embedding_dim[0] + sum(embedding_dim[3:]) + feature_dim - 1
        self.linearTraffic = nn.Linear(self.traffic_dim, attention_dim)
        #self.norm = LayerNorm(self.total_embed_dim)
        self.feed_forward = PositionwiseFeedForward((2*attention_dim+self.background_dim)*windowsz, self.output_dimension)
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
        # representation + convolution
        # [batch_sz, node2vec dim]
        # segmentEmbed = self.n2v.model(segment[:,0])
        # #print('segmentEmbed', segmentEmbed.shape)
        # # [batch_sz, window_sz, node2vec dim]
        # segmentEmbed = segmentEmbed.unsqueeze(1)
        # #print('segmentEmbed', segmentEmbed.shape)
        # segmentEmbed = torch.cat([segmentEmbed, self.n2v.model(segment[:,1]).unsqueeze(1)], dim=1)
        # segmentEmbed = torch.cat([segmentEmbed, self.n2v.model(segment[:,2]).unsqueeze(1)], dim=1)
        segmentEmbed = self.n2v(segment)
        #print('segmentEmbed', segmentEmbed.shape)
        contextual = F.relu(self.linearContextual(segmentEmbed))
        #print('contextual', contextual.shape)
        # background information time; day; mass
        background = self.embedding_time_stage(c[:, 1, :])
        #print('background', background.shape)
        background = torch.cat([background, self.embedding_week_day(c[:, 2, :]), x[:,:,1].unsqueeze(-1)], dim=-1)
        #print('background', background.shape)
        catConBack = torch.cat([contextual, background], dim=-1)
        #print('catConBack', catConBack.shape)
        #catConBack = catConBack.transpose(0, 1).contiguous()
        # [ window size, batch,  contextual+background]
        #print('catConBack', catConBack.shape)
        # [ window size, batch,  attention_dim]
        q = F.relu(self.linearQ(catConBack))
        #print('q', q.shape)

        embedded = self.embedding_road_type(c[:,0,:])
        embedded = torch.cat([embedded, self.embedding_lanes(c[:, 3, :])], dim=-1)
        embedded = torch.cat([embedded, self.embedding_bridge(c[:, 4, :])], dim=-1)
        embedded = torch.cat([embedded, self.embedding_endpoint_u(c[:, 5, :])], dim=-1)
        embedded_6 = self.embedding_endpoint_v(c[:, 6, :])
        embedded = torch.cat([embedded, embedded_6], dim=-1)
        # [ batch, window size, feature dimension+ sum embedding dimension]
        trafficPred = torch.cat([x[:,:,0].unsqueeze(-1), x[:,:,2:], embedded], dim=-1)
        #print('trafficPred', trafficPred.shape)

        # [ window size, batch,  feature dimension+ sum embedding dimension]
        #trafficPred = trafficPred.transpose(0, 1).contiguous()
        #print('trafficPred', trafficPred.shape)

        # [ batch, window size,  attention_dim]
        kv = F.relu(self.linearTraffic(trafficPred))
        #print('kv', kv.shape)

        # x -> # [ batch, window size,  attention_dim]
        x_output, output_weight = self.selfattn(q,kv,kv)
        #print('x_output', x_output.shape)

        x_output = torch.cat([catConBack,x_output], dim=-1)
        #print('x_output', x_output.shape)
        # [batch, window size*   attention_dim]
        x_output = x_output.view(x_output.shape[0],x_output.shape[1]*x_output.shape[2])
        x_output = self.feed_forward(x_output)
        #print('x_output', x_output.shape)
        return self.activate(x_output)

def testNet():
    # test nets
    net = ConstGat(n2v_dim=32, attention_dim=32, feature_dim= 6, embedding_dim=[4,2,2,2,2,4,4], num_heads=1, output_dimension=1)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    net.to(device)
    print(net)
    print(next(net.parameters()).device)
    # [batch size, window length,  feature dimension]
    tmp = torch.randn(2, 3, 6).to(device)
    # [batch, categorical_dim, window size]
    c = torch.randint(1, (2, 7, 3)).to(device)
    segments = torch.LongTensor([[465440,465440,465440],[465440,465440,465440]]).to(device)
    print(tmp.shape,c.shape,segments.shape)
    # [batch size, output dimension]
    out = net(tmp,c,segments)

    #print(net)
    #print("tmp",tmp)
    print("out", out)
    print("fc out:", out.shape)


if __name__ == "__main__":
    testNet()