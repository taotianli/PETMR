import glob
import dgl
import torch
import os
from dgl.data import DGLDataset
from loading_brain_region_data import loading_feature
from local_graph_construction import reading_brain_region
from global_graph_constuction import extract_all_global_graph_feature
from dgl.nn.pytorch import pairwise_squared_distance
import scipy.io as scio
from VGAE_train import extract_all_local_feature
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
        local_feature_dict = extract_all_local_feature()
        global_graph_dict = extract_all_global_graph_feature()
        self.graphs = []
        self.labels = []
        if global_only:
            for sub_path in glob.glob('D:/Down/Output/subjects'):
                roi_signal = global_graph_dict[sub_path][0]
                roi_signal[abs(roi_signal) < threshold] = 0
                ndata = 0
                src = []
                dst = []
                edata = []
                for i in range(r):
                    for j in range(w):
                        if roi_signal[i, j] > 0:
                            src.append(i)
                            dst.append(j)
                            edata.append([roi_signal[i, j]])
                # print(file_idx)
                graph = dgl.graph((src, dst))
                r, w = roi_signal.shape
                graph.edata['w'] = torch.tensor(edata).float()
                # print(graph.edata['w'].size())
                graph.ndata['w'] = torch.tensor(ndata[0:graph.num_nodes(), :]).float()
                # print(len(ndata))
                # print(graph.ndata['w'].size())

                self.graphs.append(graph)
                self.labels.append(torch.tensor(int(local_feature_dict[sub_path][1])))


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