import glob

import torch
import dgl
import numpy as np
from loading_brain_region_data import loading_feature
from dgl.nn.pytorch import pairwise_squared_distance
from dgl.data import CoraGraphDataset
from collections import defaultdict
from loading_brain_region_data import loading_feature
from typing import List, Tuple
import math


"""
输入：左/右脑特征矩阵
输出：每个脑区一个链接矩阵
特征按照脑区分类、计算相似性、kNN建图
采集每个vertex所属脑区，构建相关性矩阵
有几个脑区没有，有些脑区数据点很多，考虑随机采样，有些脑区很少，考虑不要了？
DGL有计算相关性的公式：无
"""


def reading_brain_region(node_feats, knn: int):
    feature = node_feats[np.argsort(node_feats[:, 3])]
    np.delete(feature, 3, axis=1)
    counter = 0
    feature_matrix_dict = dict()
    node_coor_dict = dict()
    print(feature.shape)#到底有多少个脑区，哪些脑区是没有东西的？是共性的吗？删掉unknown节点
    for i in range(37):#annot 标签从-1到35，前后保持一致
        brain_region_data = feature[counter:counter+np.sum(feature == i - 1),:]
        if i-1 in feature[:,3]:
            pass
        else:
            print(i-1)
        brain_data_torch = torch.from_numpy(brain_region_data)
        if brain_region_data.shape[0] != 0:
            knn_g = dgl.knn_graph(brain_data_torch, knn) #节点很多，应当适当增加节点数量
            knn_g.ndata['feat'] = torch.topk(pairwise_squared_distance(brain_data_torch), knn, 1, largest=False).values.to(torch.float32)
            pairwise_dists = torch.topk(pairwise_squared_distance(brain_data_torch).to(torch.float32), knn, 1, largest=False).values
            feature_matrix_dict[i] = knn_g
            node_coor_dict[i] = pairwise_dists
        else:
            pass
            # print("Brain region %d", i-1)
            # feature_matrix_dict[i] = CoraGraphDataset(reverse_edge=False)[0]
        counter += np.sum(feature == i - 1) + 1
        # print(counter)
    return feature_matrix_dict, knn_g


local_lh_feature_dict = defaultdict(list)
local_rh_feature_dict = defaultdict(list)
sub_root_path = 'D:/Down/Output/subjects/*'
for sub_path in glob.glob(sub_root_path):
    print("extract all local feature:", sub_path)
    lh_feature, rh_feature = loading_feature(sub_path)
    print('loading left brain feature.........')
    lh_data_dict, lh_data = reading_brain_region(lh_feature, knn=5)
    print('loading right brain feature...........')
    rh_data_dict, rh_data = reading_brain_region(rh_feature, knn=5)

# root_path = 'D:/Down/Output/subjects/sub-02'
# lh_feature, rh_feature = loading_feature(root_path)
# lh_feature_dict, kg = reading_brain_region(lh_feature, knn=5)