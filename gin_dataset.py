import glob
import dgl
import torch
import os
from dgl.data import DGLDataset
from loading_brain_region_data import loading_feature
from local_graph_construction import reading_brain_region
from global_graph_constuction import concat_global_local_graph, concat_global_graph, loading_global_graph
from dgl.nn.pytorch import pairwise_squared_distance
import scipy.io as scio
from dgl import save_graphs, load_graphs
from sklearn.metrics import confusion_matrix


class MyDataset(DGLDataset):
    def __init__(self,
                 url=None,
                 raw_dir=None,
                 save_dir=None,
                 force_reload=False,
                 verbose=False):
        super(MyDataset, self).__init__(name='dataset_name',
                                        url=url,
                                        raw_dir=raw_dir,
                                        save_dir=save_dir,
                                        force_reload=force_reload,
                                        verbose=verbose)

    def process(self):
        root_path = 'D:/Down/Output/subjects/sub-02'
        lh_feature, rh_feature = loading_feature(root_path)
        lh_feature_dict, kg = reading_brain_region(lh_feature, knn=5)
        global_feas = loading_global_graph(root_path)
        global_matrix = torch.topk(pairwise_squared_distance(global_feas).to(torch.float32), 5, 1, largest=False).values
        glo_graph = concat_global_local_graph(global_matrix, global_feas, 1)
        self.graphs = []
        self.labels = []
        if global_only:
            for i in lh_feature_dict:

                for root_path in glob.glob('D:/Down/Output/subjects/'):
                    file_name = data[root_path].replace('\n', '')
                    if os.path.exists(file_name):
                        mat_data = scio.loadmat(file_name)
                        roi_signal = mat_data['SignalMatrix2d']
                        roi_signal[abs(roi_signal) < threshold] = 0

                        ndata = mat_data['SignalMatrix3d']
                        # print(ndata[0:22,:].shape)
                        r, w = roi_signal.shape

                        src = []
                        dst = []
                        edata = []
                        # print(file_name)
                        for i in range(r):
                            for j in range(w):
                                if roi_signal[i, j] > 0:
                                    src.append(i)
                                    dst.append(j)
                                    edata.append([roi_signal[i, j]])
                        # print(file_idx)
                        graph = dgl.graph((src, dst))
                        # print(len(src),len(dst))
                        # print(graph.num_nodes())

                        graph.edata['w'] = torch.tensor(edata).float()
                        # print(graph.edata['w'].size())
                        graph.ndata['w'] = torch.tensor(ndata[0:graph.num_nodes(), :]).float()
                        # print(len(ndata))
                        # print(graph.ndata['w'].size())

                        self.graphs.append(graph)
                        self.labels.append(torch.tensor(int(mat_data['Signal_label'])))

        else:
            pass



    def __getitem__(self, idx):
        return self.graphs[idx], self.labels[idx]

    def __len__(self):
        return len(self.graphs)


threshold = 0.8
global_only = True
os.chdir("D:\CHB seizure\GIH data")
with open("D:\CHB seizure\GIH data\path.txt", "r") as f:
    data = f.readlines()
dataset = MyDataset()