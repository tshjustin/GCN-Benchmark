# GCN-Benchmark
SC4020 Project on GCN Algorithm Benchmarking 

### Setting up Environment 
```
py -3.8 -m venv venv
venv/Scripts/Activate 
pip install -r requirements.txt 
```

### GCN / GAT Model Details

|                    | Description        |
| -------------------|------------------|
| Model Architecture | n-GCN / Attention Layer| 
| Loss Function      | CrossEntropy Loss  |  
| Optimizer          | Adam Optimizer     | 
| Evaluation Metric | Accuracy |


### Arguments 
```
--num_heads: number of attention heads for GAT

--num_layers: number of training layers / amount of propogation 
--hidden_dim: number of hidden dimensions 
--dropout: ratio to dropout to prevent overfitting 
--use-bias: bias to balance 


--lr : learning rate 
--weight_decay: decay rate of learning rate 
--epochs: training loops 

Example Usage with CLI: 
python train.py --model GAT --dataset CORA --num_heads 4 8 --num_layers 2 3 --hidden_dim 16 32 --dropout 0.5 --lr 0.001
```