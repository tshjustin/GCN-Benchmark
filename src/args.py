from argparse import ArgumentParser

parser = ArgumentParser()

# Model choice
parser.add_argument("--model", type=str, choices=["GAT", "GCN"], default="GAT")

# Dataset choice
parser.add_argument("--dataset", type=str, choices=["SQU", "CORA"], default="CORA")

# Data Parser 
parser.add_argument("--nodes_path", type=str, default="cora/cora.content")
parser.add_argument("--edges_path", type=str, default="cora/cora.cites")

# GAT-Specific Parameters
parser.add_argument("--n_heads", type=int, default=8)
parser.add_argument("--n_hidden_units", type=int, default=8)
parser.add_argument("--leaky_relu_slope", type=float, default=0.2)
parser.add_argument("--concat", type=bool, default=False)

# Model Parameters (For both GAT & GCN)
parser.add_argument("--hidden_dim", type=int, default=16)
parser.add_argument("--dropout", type=float, default=0.5)
parser.add_argument("--use-bias", type=bool, default=True)
parser.add_argument("--num_layers", type=int, default=2)

# Dataset Parameters  
parser.add_argument("--train_proportion", type=float, default=0.6)
parser.add_argument("--validation_proportion", type=float, default=0.2)
parser.add_argument("--test_proportion", type=float, default=0.2)

# Training Parameters 
parser.add_argument("--lr", type=float, default=1e-2)
parser.add_argument("--weight_decay", type=float, default=5e-3)
parser.add_argument("--patience", type=int, default=10)
parser.add_argument("--epochs", type=int, default=200)
parser.add_argument("--early_termination", type=bool, default=True)

config = parser.parse_args()

if config.dataset == "CORA":
    config.nodes_path = "cora/cora.content"
    config.edges_path = "cora/cora.cites"

elif config.dataset == "SQU":
    config.nodes_path = "squirrel/out1_node_feature_label.txt"
    config.edges_path = "squirrel/out1_graph_edges.txt"