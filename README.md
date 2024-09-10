# GCN-Benchmark
SC4020 Project on GCN Algorithm Benchmarking 

### Project Details 

|                    | Description        |
| -------------------|:------------------:|
| Model Architecture | 2-GCN Layer        | 
| Loss Function      | CrossEntropy Loss  |  
| Optimizer          | Adam Optimizer     | 


### Setting up Environment 
```
py -3.8 -m venv venv
venv/Scripts/Activate 
pip install -r requirements.txt 
```

### Things to add 
1. Trying out different parameters  (Learning rates, etc)

2. Add visuals of performances 


### File Structure 

├── cora/
|     |── README 
|     ├── cora.cites   # Citation relationships    
│     ├── cora.content           # Node feature and label data            
│   
├── src/
│   ├── args.py                # argument parser configs
│   ├── dataloader.py          # load dataset (train_test_split etc)
│   ├── evaluation.py          # evaluation 
│   ├── main.py                # entry point 
│   ├── model.py               # model definition 
│   ├── utils.py               # utility functions 
├── EDA_CORA.ipynb             # EDA on CORA
├── EDA_REDDIT.ipynb           # EDA on REDDIT 
