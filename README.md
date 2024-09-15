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


### CORA Data 
The Cora dataset consists of 2708 scientific publications classified into one of seven classes. The citation network consists of 5429 links. Each publication in the dataset is described by a 0/1-valued word vector indicating the absence/presence of the corresponding word from the dictionary. The dictionary consists of 1433 unique words.