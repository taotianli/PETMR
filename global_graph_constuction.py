from sklearn import preprocessing
import pandas as pd
import glob
import numpy as np
import torch
import dgl
from collections import defaultdict
from dgl.nn.pytorch import pairwise_squared_distance


def loading_global_graph(root_path):
    area_file_path = glob.glob(root_path + '/ses-M00/t1/freesurfer_cross_sectional/regional_measures/*desikan_area*')[0]
    area_data = preprocessing.scale(pd.read_csv(area_file_path, sep='\t')['label_value'][0:68])
    meancurv_file_path = glob.glob(root_path + '/ses-M00/t1/freesurfer_cross_sectional/regional_measures/*desikan_meancurv*')[0]
    meancurv_data = pd.read_csv(meancurv_file_path, sep='\t')['label_value'][0:68]
    thickness_file_path = glob.glob(root_path + '/ses-M00/t1/freesurfer_cross_sectional/regional_measures/*desikan_thickness*')[0]
    thickness_data = pd.read_csv(thickness_file_path, sep='\t')['label_value'][0:68]
    volume_file_path = glob.glob(root_path + '/ses-M00/t1/freesurfer_cross_sectional/regional_measures/*desikan_volume*')[0]
    volume_data = pd.read_csv(volume_file_path, sep='\t')['label_value'][0:68]
    pet_file_path = glob.glob(root_path + '/ses-M00/pet/surface/atlas_statistics/*desi*')[0]
    pet_data = pd.read_csv(pet_file_path, sep='\t')['mean_scalar']
    pet_data.drop([0, 1, 8, 9], inplace=True)
    pet_data = pet_data[0:68]
    global_feature = np.c_[area_data, meancurv_data, thickness_data, volume_data, pet_data]
    return global_feature


def concat_graph(graph, feature):
    knn_g = dgl.knn_graph(feature, 5)
    knn_g.ndata['feat'] = feature
    return knn_g


def extract_all_global_graph_feature(sub_root_path='D:/Down/Output/subjects/*', knn_node_num=5):
    print('Loading global graph data!')
    global_graph_dict = defaultdict(list)
    for sub_path in glob.glob(sub_root_path):
        global_feas = torch.from_numpy(loading_global_graph(sub_path))
        global_sturc = torch.topk(pairwise_squared_distance(global_feas).to(torch.float32),
                   knn_node_num, 1, largest=False).values
        global_graph = concat_graph(global_sturc, global_feas)
        global_graph_dict[sub_path.split('\\')[1]].append(global_feas)
        global_graph_dict[sub_path.split('\\')[1]].append(global_graph)
    return global_graph_dict



# root_path = 'D:/Down/Output/subjects/sub-02'
# global_feas = loading_global_graph(root_path)
# global_matrix = torch.topk(pairwise_squared_distance(global_feas).to(torch.float32), 5, 1, largest=False).values
# glo_graph = concat_global_local_graph(global_matrix, global_feas, 1)
