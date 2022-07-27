import dgl
import torch
import os
from dgl.data import DGLDataset
import scipy.io as scio
from dgl import save_graphs, load_graphs
from sklearn.metrics import confusion_matrix


class MyDataset(DGLDataset):
    def __init__(self,
                 url=None,
                 raw_dir=None,
                 save_dir=None,
                 force_reload=False,
                 verbose=False,
                 global_only=True):
        super(MyDataset, self).__init__(name='dataset_name',
                                        url=url,
                                        raw_dir=raw_dir,
                                        save_dir=save_dir,
                                        force_reload=force_reload,
                                        verbose=verbose)

    def process(self):

        self.graphs = []
        self.labels = []


    def __getitem__(self, idx):
        return self.graphs[idx], self.labels[idx]

    def __len__(self):
        return len(self.graphs)


threshold = 0.8
os.chdir("D:\CHB seizure\GIH data")
with open("D:\CHB seizure\GIH data\path.txt", "r") as f:
    data = f.readlines()
dataset = MyDataset()