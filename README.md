# GCN-Benchmark
SC4020 Project on GCN Algorithm Benchmarking 

### Setting up Environment 
```
py -3.8 -m venv venv
venv/Scripts/Activate 
pip install -r requirements.txt 
```

### Model Details 

|                    | Description        |
| -------------------|:------------------:|
| Model Architecture | n-GCN Layer        | 
| Loss Function      | CrossEntropy Loss  |  
| Optimizer          | Adam Optimizer     | 

### Hyperparameters 
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

### CORA Data 
The Cora dataset consists of 2708 scientific publications classified into one of seven classes. The citation network consists of 5429 links. Each publication in the dataset is described by a 0/1-valued word vector indicating the absence/presence of the corresponding word from the dictionary. The dictionary consists of 1433 unique words.