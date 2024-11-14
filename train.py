import copy

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

import config
from directed_train_test_split import direction_specific, direction_blind
from model import ADI_MSF

edge_list = np.loadtxt('./newdataset/edgelist.txt', dtype=np.int64)
edge_index = torch.tensor(edge_list).t().contiguous()
num_nodes = len(set(edge_index.flatten().tolist()))

feature = pd.read_csv('./newdataset/1752morgan1024.csv', header=0, index_col=0)
features = torch.from_numpy(feature.values).to(torch.float32)



def train():
    model.train()
    optimizer.zero_grad()
    z_in, z_out, z_self, z_self_re = model.encode(x, train_pos_edge_index)
    loss = model.recon_loss(z_in, z_out, z_self, x, z_self_re, train_pos_edge_index, task1=config.task1)
    loss.backward()
    optimizer.step()
    return float(loss)


def test(pos_edge_index, neg_edge_index):
    model.eval()
    with torch.no_grad():
        z_in, z_out, z_self, _ = model.encode(x, train_pos_edge_index)
    return model.test(z_in, z_out, z_self, pos_edge_index, neg_edge_index)


def testfinal(pos_edge_index, neg_edge_index):
    model.eval()
    with torch.no_grad():
        z_in, z_out, z_self, _ = model.encode(x, train_pos_edge_index)
    return model.test(z_in, z_out, z_self, pos_edge_index, neg_edge_index)


def initialize_list():
    lists = [[] for _ in range(6)]
    return [lists[i] for i in range(6)]


target = ["auc", "ap", "acc", "f1", "pre", "re"]
auc_list, ap_list, f1_list, acc_list, pre_list, re_list = initialize_list()
auc_std_list, ap_std_list, f1_std_list, acc_std_list, pre_std_list, re_std_list = initialize_list()
target_list = [auc_list, ap_list, f1_list, acc_list, pre_list, re_list]
target_std_list = []




for i in range(config.number):

    auc_l, ap_l, f1_l, acc_l, pre_l, re_l = initialize_list()
    target_l = [auc_l, ap_l, f1_l, acc_l, pre_l, re_l]
    for fold in range(1, config.fold + 1):
        data = Data(edge_index=edge_index, num_nodes=num_nodes, x=features)
        if config.task1:
            data = direction_specific(data, fold, config.seed)
        else:
            data = direction_blind(data, fold, config.seed)
        print(data)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        train_pos_edge_index = data.train_pos_edge_index.to(device)
        x = data.x.to(device)
        model = ADI_MSF(data.num_node_features, config.out_channels).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

        min_loss_val = config.min_loss_val
        best_model = None
        best_epoch = 0
        min_epoch = config.min_epoch
        best_f1 = 0
        for epoch in range(1, config.epochs + 1):
            loss = train()
            auc, ap, acc, f1, pre, re = test(data.val_pos_edge_index, data.val_neg_edge_index)
            if epoch % 10 == 0:
                print('Epoch: {:03d} AUC: {:.4f} AP: {:.4f} ACC: {:.4f} F1: {:.4f} PRE: {:.4f} RE: {:.4f} Loss: {:.4f}'
                      .format(epoch, auc, ap, acc, f1, pre, re, loss))
            if f1 >= best_f1:
                best_f1 = f1
                best_model = copy.deepcopy(model)
                best_epoch = epoch
        model = best_model
        auc, ap, acc, f1, pre, re = testfinal(data.test_pos_edge_index, data.test_neg_edge_index)
        # save_emb(data.test_pos_edge_index,fold)
        print(
            'Fold_{}_Test_BestEpoch_{}. AUC: {:.4f}, AP: {:.4f}, ACC: {:.4f}, F1: {:.4f}, PRE: {:.4f}, RE: {:.4f}'
            .format(fold, best_epoch, auc, ap, acc, f1, pre, re))

        for j in range(6):
            target_l[j].append(eval(target[j]))
    for j in range(6):
        target_list[j].append(np.mean(target_l[j]))
print('-' * 20, 'Average_Final. ', '-' * 20)
print(
    'AUROC: {:.4f} AUPRC: {:.4f}  ACC: {:.4f}  F1: {:.4f}  PRECISION:{:.4f}  RECALL: {:.4f} '
    .format(np.mean(target_list[0]),
            np.mean(target_list[1]),
            np.mean(target_list[2]),
            np.mean(target_list[3]),
            np.mean(target_list[4]),
            np.mean(target_list[5])))
print('-' * 20, 'STD. ', '-' * 20)

print(
    'AUROC: ±{:.4f} AUPRC: ±{:.4f}  ACC: ±{:.4f}  F1: ±{:.4f}  PRECISION:±{:.4f}  RECALL: ±{:.4f} '
    .format(np.std(target_list[0]),
            np.std(target_list[1]),
            np.std(target_list[2]),
            np.std(target_list[3]),
            np.std(target_list[4]),
            np.std(target_list[5])))
