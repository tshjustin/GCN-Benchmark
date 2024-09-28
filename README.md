# GCN-Benchmark
SC4020 Project on GCN Algorithm Benchmarking 

The 3 Datasets of choice are: 

| Dataset                   | Description        |
| -------------------|------------------|
| Cora | Data is downloaded straight from source. Source: https://linqs.org/datasets/      | 
| PPI      | Inductive PPI data (One big Graph) https://github.com/tshjustin/PPI_EDA |  
| PPI | Transductive PPI data (24 Graphs) https://github.com/tshjustin/PPI_Inductive_EDA |
| Pubmed          |      | 

### CORA Data 
The Cora dataset consists of 2708 scientific publications classified into one of seven classes. The citation network consists of 5429 links. Each publication in the dataset is described by a 0/1-valued word vector indicating the absence/presence of the corresponding word from the dictionary. The dictionary consists of 1433 unique words.


### PPI Data 
The PPI dataset consist of 24 Graphs - Split into 20 Train , 2 Validation and 2 Test. Each Nodes represents proteins and edges  represents a molecular bond between each protein. Each node in the dataset is described by a 50-vector feauture that describes the protein. There are a total of 121 possible labels, and each node may be classified with multiple labels.


### Setting up Environment 
```
py -3.8 -m venv venv
venv/Scripts/Activate 
pip install -r requirements.txt 
```

### GCN / GAT Model Details - CORA  / Squirrel

|                    | Description        |
| -------------------|------------------|
| Model Architecture | n-GCN / Attention Layer| 
| Loss Function      | CrossEntropy Loss  |  
| Optimizer          | Adam Optimizer     | 
| Evaluation Metric | Accuracy |

### GCN / GAT Model Details - PPI 

|                    | Description        |
| -------------------|------------------|
| Model Architecture | n-GCN / Attention Layer        | 
| Loss Function      | Binary Cross-Entropy Loss  |  
| Optimizer          | Adam Optimizer     | 
| Evaluation Metric | F1 Score | 

### Arguments 
```
--num_heads: number of attention heads for GAT

--hidden_dim: number of hidden dimensions 
--dropout: ratio to dropout to prevent overfitting 
--use-bias: bias to balance 
--num_layers: number of training layers / amount of propogation 

--lr : learning rate 
--weight_decay: decay rate of learning rate 
--epochs: training loops 

Example Usage with CLI: 
python train.py --model GAT --dataset CORA --num_heads 4 8 --num_layers 2 3 --hidden_dim 16 32 --dropout 0.5 --lr 0.001
```