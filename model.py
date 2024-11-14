import torch
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, f1_score, recall_score, \
    precision_score
from torch_geometric.nn import GATConv
from torch_geometric.nn.inits import reset
from torch_geometric.utils import (negative_sampling)

import config


class ADI_MSF(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(ADI_MSF, self).__init__()
        self.encoder = Encoder(in_channels, out_channels)
        self.decoder = Decoder(out_channels * 3)
        ADI_MSF.reset_parameters(self)

    def encode(self, *args, **kwargs):
        return self.encoder(*args, **kwargs)

    def decode(self, *args, **kwargs):
        return self.decoder(*args, **kwargs)

    def reset_parameters(self):
        reset(self.encoder)
        reset(self.decoder)

    def recon_loss(self, z_in, z_out, z_self, x, z_self_re, pos_edge_index, task1=True):
        pos_loss = -torch.log(self.decoder(z_in, z_out, z_self, pos_edge_index, sigmoid=True) + config.EPS).mean()

        if task1:
            neg_edge_index = negative_sampling(pos_edge_index, z_self.size(0))
        else:
            neg_edge_index = torch.stack([pos_edge_index[1], pos_edge_index[0]], dim=0).cuda()
        neg_loss = -torch.log(1 - self.decoder(z_in, z_out, z_self, neg_edge_index, sigmoid=True) + config.EPS).mean()
        mse_loss = nn.MSELoss(reduction='mean')
        return config.alpha * (pos_loss + neg_loss) + (1 - config.alpha) * mse_loss(z_self_re, x)

    def test(self, z_in, z_out, z_self, pos_edge_index, neg_edge_index):
        pos_y = z_self.new_ones(pos_edge_index.size(1))
        neg_y = z_self.new_zeros(neg_edge_index.size(1))
        y = torch.cat([pos_y, neg_y], dim=0)
        pos_pred = self.decoder(z_in, z_out, z_self, pos_edge_index, sigmoid=True)
        neg_pred = self.decoder(z_in, z_out, z_self, neg_edge_index, sigmoid=True)
        pred = torch.cat([pos_pred, neg_pred], dim=0)
        y, pred = y.detach().cpu().numpy(), pred.detach().cpu().numpy()

        return roc_auc_score(y, pred), average_precision_score(y, pred), accuracy_score(y, pred.round()), \
            f1_score(y, pred.round()), precision_score(y, pred.round()), recall_score(y, pred.round())


class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Encoder, self).__init__()
        self.graph_in_encoder = GATEncoder(in_channels, out_channels, flow='source_to_target')
        self.graph_out_encoder = GATEncoder(in_channels, out_channels, flow='target_to_source')
        self.self_feat_encoder = AutoEncoder(in_channels, out_channels)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, edge_index):
        # F.dropout(x, p=0.6, training=self.training)
        x_in = self.graph_in_encoder(x, edge_index)
        x_in = self.dropout(x_in)
        x_out = self.graph_out_encoder(x, edge_index)
        x_out = self.dropout(x_out)
        x_self, z_self_re = self.self_feat_encoder(x)
        x_self = self.dropout(x_self)
        return x_in, x_out, x_self, z_self_re


class Decoder(nn.Module):
    def __init__(self, in_channels):
        super(Decoder, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, in_channels // 2),
            nn.ELU(),
            nn.Dropout(0.1),
            nn.Linear(in_channels // 2, 1),
        )

    def forward(self, z_in, z_out, z_self, edge_index, sigmoid=True):
        v1 = z_out + z_self
        v2 = z_in + z_self

        value = v1[edge_index[0]] * v2[edge_index[1]]  # Hadamard

        output = self.mlp(value)

        return torch.sigmoid(output) if sigmoid else output


class AutoEncoder(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(AutoEncoder, self).__init__()
        self.encoder1 = nn.Sequential(
            nn.Linear(in_dim, out_dim * 2),
            nn.ELU(),
            nn.BatchNorm1d(out_dim * 2),
            nn.Dropout(0.1)
        )

        self.encoder2 = nn.Sequential(
            nn.Linear(out_dim * 2, out_dim),
            nn.ELU(),
            nn.BatchNorm1d(out_dim),
        )
        self.decoder1 = nn.Linear(out_dim, out_dim * 2)
        self.decoder2 = nn.Linear(out_dim * 2, in_dim)

    def forward(self, x):
        z_1 = self.encoder1(x)
        z_2 = self.encoder2(z_1)
        z_emb = torch.cat((z_1, z_2), dim=-1)

        d1 = F.elu(self.decoder1(z_2))
        d_emb = torch.sigmoid(self.decoder2(d1))

        return z_emb, d_emb


class GATEncoder(nn.Module):
    def __init__(self, in_dim, out_dim, flow):
        super(GATEncoder, self).__init__()
        self.gat1 = GATConv(in_dim, out_dim * 2, flow=flow,
                            heads=32, concat=False, add_self_loops=False)
        self.gat2 = GATConv(out_dim * 2, out_dim, flow=flow,
                            heads=32, concat=False, add_self_loops=False)

    def forward(self, x, edge_index):
        z1 = F.elu(self.gat1(x, edge_index))
        z2 = F.elu(self.gat2(z1, edge_index))

        return torch.cat((z1, z2), dim=-1)
