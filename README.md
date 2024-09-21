# GCN-Benchmark
SC4020 Project on GCN Algorithm Benchmarking 

The 3 Datasets of choice are: 

| Dataset                   | Description        |
| -------------------|------------------|
| Cora | Data is downloaded straight from source. Source: https://linqs.org/datasets/      | 
| PPI      |  Data is cleaned from an external source. Cleaning Code can be found: https://github.com/tshjustin/PPI_EDA |  
| Pubmed          |      | 

### CORA Data 
The Cora dataset consists of 2708 scientific publications classified into one of seven classes. The citation network consists of 5429 links. Each publication in the dataset is described by a 0/1-valued word vector indicating the absence/presence of the corresponding word from the dictionary. The dictionary consists of 1433 unique words.


### PPI Data 
The PPI dataset consist of 56944 Nodes which represents proteins and and 818217 Edges that represents a molecular bond between each protein. Each node in the dataset is described by a 50-vector feauture that describes the protein. There are a total of 121 possible labels, and each node may be classified with multiple labels (Rather than 1)


### Setting up Environment 
```
py -3.8 -m venv venv
venv/Scripts/Activate 
pip install -r requirements.txt 
```

### GCN Model Details 

|                    | Description        |
| -------------------|:------------------:|
| Model Architecture | n-GCN Layer        | 
| Loss Function      | CrossEntropy Loss  |  
| Optimizer          | Adam Optimizer     | 

### Hyperparameters of GCN 
```
--nodes_path : path to nodes 
--edges_path: path to edges 
--hidden_dim: number of hidden dimensions 
--dropout: ratio to dropout to prevent overfitting 
--use-bias: bias to balance 

--num_layers: number of training layers / amount of propogation 
--train_proportion
--validation_proportion
--test_proportion

--lr : learning rate 
--weight_decay: decay rate of learning rate 
--patience: number of runs where improvement less than 0.01 before terminating 
--epochs: training loops 
--early_termination: if patience number is hit, then early terminates 
```

