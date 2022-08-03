import argparse
import glob
import os
import time

import dgl
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
import torch
import torch.nn.functional as F

import model
from preprocess import mask_test_edges, mask_test_edges_dgl, sparse_to_tuple, preprocess_graph
from loading_brain_region_data import loading_feature, loading_label
from local_graph_construction import reading_brain_region
from collections import defaultdict

"""
重构函数 MyDataset、DataLoader
"""


os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

parser = argparse.ArgumentParser(description='Variant Graph Auto Encoder')
parser.add_argument('--learning_rate', type=float, default=0.01, help='Initial learning rate.')
parser.add_argument('--epochs', '-e', type=int, default=3, help='Number of epochs to train.')
parser.add_argument('--hidden1', '-h1', type=int, default=32, help='Number of units in hidden layer 1.')
parser.add_argument('--hidden2', '-h2', type=int, default=16, help='Number of units in hidden layer 2.')
parser.add_argument('--datasrc', '-s', type=str, default='dgl',
                    help='Dataset download from dgl Dataset or website.')
parser.add_argument('--dataset', '-d', type=str, default='cora', help='Dataset string.')
parser.add_argument('--gpu_id', type=int, default=0, help='GPU id to use.')
args = parser.parse_args()


# check device
# device = torch.device("cuda:{}".format(args.gpu_id) if torch.cuda.is_available() else "cpu")
device = "cpu"


def compute_loss_para(adj):
    pos_weight = ((adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum())
    norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)
    weight_mask = adj.view(-1) == 1
    weight_tensor = torch.ones(weight_mask.size(0)).to(device)
    weight_tensor[weight_mask] = pos_weight
    return weight_tensor, norm


def get_acc(adj_rec, adj_label):
    labels_all = adj_label.view(-1).long()
    preds_all = (adj_rec > 0.5).view(-1).long()
    accuracy = (preds_all == labels_all).sum().float() / labels_all.size(0)
    return accuracy


def get_scores(edges_pos, edges_neg, adj_rec):
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    adj_rec = adj_rec.cpu()
    # Predict on test set of edges
    preds = []
    for e in edges_pos:
        preds.append(sigmoid(adj_rec[e[0], e[1]].item()))

    preds_neg = []
    for e in edges_neg:
        preds_neg.append(sigmoid(adj_rec[e[0], e[1]].data))

    preds_all = np.hstack([preds, preds_neg])
    labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds_neg))])
    roc_score = roc_auc_score(labels_all, preds_all)
    ap_score = average_precision_score(labels_all, preds_all)
    return roc_score, ap_score


def dgl_main(data):
    # Load from DGL dataset
    graph = data

    # Extract node features
    feats = graph.ndata.pop('feat').to(device)
    in_dim = feats.shape[-1]
    graph = graph.to(device)

    # create model
    vgae_model = model.VGAEModel(in_dim, args.hidden1, args.hidden2)
    vgae_model = vgae_model.to(device)

    # create training component
    optimizer = torch.optim.Adam(vgae_model.parameters(), lr=args.learning_rate)
    print('Total Parameters:', sum([p.nelement() for p in vgae_model.parameters()]))

    # create training epoch
    for epoch in range(args.epochs):
        vgae_model.train()
        logits, hg = vgae_model.forward(graph, feats)
        optimizer.zero_grad()
        optimizer.step()
    return hg


def extract_all_local_feature(sub_root_path='D:/Down/Output/subjects/*'):
    print("Starting loading local feature!")
    local_lh_feature_dict = defaultdict(list)
    local_rh_feature_dict = defaultdict(list)
    sub_info = loading_label()
    for sub_path in glob.glob(sub_root_path):
        print("extract all local feature:", sub_path)
        temp_single_sub_lh_feature = list()
        temp_single_sub_rh_feature = list()
        lh_feature, rh_feature = loading_feature(sub_path)
        lh_data_dict, lh_data = reading_brain_region(lh_feature, knn=5)
        rh_data_dict, rh_data = reading_brain_region(rh_feature, knn=5)
        for i in range(35):
            print(i, end=' ')
            sampled_lh_z = dgl_main(lh_data_dict[i])
            temp_single_sub_lh_feature.append(sampled_lh_z)
            sampled_rh_z = dgl_main(rh_data_dict[i])
            temp_single_sub_rh_feature.append(sampled_rh_z)
        local_lh_feature_dict[sub_path] = temp_single_sub_lh_feature
        local_rh_feature_dict[sub_path] = temp_single_sub_rh_feature
        # local_feature_dict[sub_path] = sub_info[sub_path]
        print('One Subject Finished!')
    return local_lh_feature_dict
