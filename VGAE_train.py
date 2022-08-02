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
    # if args.dataset == 'cora':
    #     dataset = CoraGraphDataset(reverse_edge=False)
    # elif args.dataset == 'citeseer':
    #     dataset = CiteseerGraphDataset(reverse_edge=False)
    # elif args.dataset == 'pubmed':
    #     dataset = PubmedGraphDataset(reverse_edge=False)
    # else:
    #     raise NotImplementedError
    graph = data
    # for k,v in dataset_dict:
    #     graph = dataset_dict[0]

    # Extract node features
    feats = graph.ndata.pop('feat').to(device)
    in_dim = feats.shape[-1]

    # generate input
    adj_orig = graph.adjacency_matrix().to_dense()

    # build test set with 10% positive links
    train_edge_idx, val_edges, val_edges_false, test_edges, test_edges_false = mask_test_edges_dgl(graph, adj_orig)

    graph = graph.to(device)
    # print(graph)

    # create train graph
    train_edge_idx = torch.tensor(train_edge_idx).to(device)
    train_graph = dgl.edge_subgraph(graph, train_edge_idx, preserve_nodes=True)
    train_graph = train_graph.to(device)
    # print(train_graph)
    adj = train_graph.adjacency_matrix().to_dense().to(device)
    # print(adj.size())

    # compute loss parameters
    weight_tensor, norm = compute_loss_para(adj)

    # create model
    vgae_model = model.VGAEModel(in_dim, args.hidden1, args.hidden2)
    vgae_model = vgae_model.to(device)

    # create training component
    optimizer = torch.optim.Adam(vgae_model.parameters(), lr=args.learning_rate)
    print('Total Parameters:', sum([p.nelement() for p in vgae_model.parameters()]))

    # create training epoch
    for epoch in range(args.epochs):
        t = time.time()

        # Training and validation using a full graph
        vgae_model.train()

        logits, hg = vgae_model.forward(graph, feats)
        print("this is AE feature", hg)

        # compute loss
        print(logits.view(-1).size(), adj.view(-1).size(),adj.size(),logits.size())
        loss = norm * F.binary_cross_entropy(logits.view(-1), adj.view(-1), weight=weight_tensor)
        # print(loss)
        kl_divergence = 0.5 / logits.size(0) * (
                1 + 2 * vgae_model.log_std - vgae_model.mean ** 2 - torch.exp(vgae_model.log_std) ** 2).sum(
            1).mean()
        loss -= kl_divergence

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_acc = get_acc(logits, adj)

        val_roc, val_ap = get_scores(val_edges, val_edges_false, logits)

        # Print out performance
        print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(loss.item()), "train_acc=",
              "{:.5f}".format(train_acc), "val_roc=", "{:.5f}".format(val_roc), "val_ap=", "{:.5f}".format(val_ap),
              "time=", "{:.5f}".format(time.time() - t))

    test_roc, test_ap = get_scores(test_edges, test_edges_false, logits)
    # roc_means.append(test_roc)
    # ap_means.append(test_ap)
    print("End of training!", "test_roc=", "{:.5f}".format(test_roc), "test_ap=", "{:.5f}".format(test_ap))
    return hg


#
# if __name__ == '__main__':
#     dgl_main()
def extract_all_local_feature(sub_root_path='D:/Down/Output/subjects/*'):
    # root_path = 'D:/Down/Output/subjects/sub-02'
    # lh_feature, rh_feature = loading_feature(root_path)
    # dataset_dict, sub_data = reading_brain_region(lh_feature, knn=5)
    # dgl_main(sub_data)
    print("Starting loading local feature!")
    local_feature_dict = defaultdict(list)
    sub_info = loading_label()
    for sub_path in glob.glob(sub_root_path):
        print("extract all local feature:", sub_path)
        temp_single_sub_feature = list()
        lh_feature, rh_feature = loading_feature(sub_path)
        sub_data_dict, sub_data = reading_brain_region(lh_feature, knn=5)
        for i in range(35):
            sampled_z = dgl_main(sub_data_dict[i])
            temp_single_sub_feature.append(sampled_z)
        local_feature_dict[sub_path] = temp_single_sub_feature
        # local_feature_dict[sub_path] = sub_info[sub_path]
        print('One Subject Finished!')
    return local_feature_dict
