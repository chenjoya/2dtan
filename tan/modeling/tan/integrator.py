import torch
from torch import nn
from torch.functional import F

class Integrator(nn.Module):
    def __init__(self, feat_hidden_size, query_input_size, query_hidden_size, 
            bidirectional, num_layers):
        super(Integrator, self).__init__()
        if bidirectional:
            query_hidden_size //= 2
        self.lstm = nn.LSTM(
            query_input_size, query_hidden_size, num_layers=num_layers, 
            bidirectional=bidirectional, batch_first=True
        )
        self.fc = nn.Linear(query_hidden_size, feat_hidden_size)
        self.conv = nn.Conv2d(feat_hidden_size, feat_hidden_size, 1, 1)

    def encode_query(self, queries, wordlens):
        self.lstm.flatten_parameters()
        queries = self.lstm(queries)[0] 
        queries = queries[range(queries.size(0)), wordlens.long() - 1]
        return self.fc(queries)

    def forward(self, queries, wordlens, map2d):
        queries = self.encode_query(queries, wordlens)[:,:,None,None]
        map2d = self.conv(map2d)
        # print('map2d.shape',map2d.shape)
        return F.normalize(queries * map2d)

def build_integrator(cfg):
    feat_hidden_size = cfg.MODEL.TAN.FEATPOOL.HIDDEN_SIZE 
    query_input_size = cfg.INPUT.PRE_QUERY_SIZE
    query_hidden_size = cfg.MODEL.TAN.INTEGRATOR.QUERY_HIDDEN_SIZE 
    bidirectional = cfg.MODEL.TAN.INTEGRATOR.LSTM.BIDIRECTIONAL
    num_layers = cfg.MODEL.TAN.INTEGRATOR.LSTM.NUM_LAYERS
    return Integrator(feat_hidden_size, query_input_size, query_hidden_size, 
        bidirectional, num_layers) 
