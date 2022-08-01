### Multiview Cortical Graph Isomorphism Network (MC-GIN) for MCI/NC classification based on PET/MR sync data


## Getting Started

The whole project consists two parts: Clinica Surface Processing and Graph Neural Network construction (VGAE and GIN).



### Installing
The conda environment is as follows

```
dgl=0.9.0
nibabel=4.0.1
pytorch=1.10.2
scikit-learn=1.0.2
scipy=1.7.3
```
Quick distribution: 

```
conda install --yes --file requirements.txt
```

## Results

Use *area under the ROC curve* (AUC) and *average precision* (AP) scores for each model on the test set. Numbers show mean results and standard error for 10 runs with random initializations on fixed dataset splits.


| Method | ACC | SEN | SPE | AUC |
|--------|-----|-----|-----|-----|
| SVM    |     |     |     |     |
| GCN    |     |     |     |     |
| GAT    |     |     |     |     |
| g-GIN  |     |     |     |     |
| l-GIN  |     |     |     |     |
| MC-GIN |     |     |     |     |




## Built With

* [Clinica](http://www.clinica.run/) - The web framework used
* [DGL](https://github.com/dmlc/dgl/) - Main framework


## Authors
* **Tianli Tao @ShanghaiTech University** - *Initial work* 
