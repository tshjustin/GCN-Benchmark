from argparse import ArgumentParser

parser = ArgumentParser()

# Model choice
parser.add_argument("--model", type=str, choices=["GAT", "GCN"], default="GCN")

# Dataset choice
parser.add_argument("--dataset", type=str, choices=["SQU", "CORA"], default="SQU")

# For GAT 
parser.add_argument("--n_heads", type=int, default=8)

# Model Parameters (For both GAT & GCN)
parser.add_argument("--num_layers", type=int, default=2)
parser.add_argument("--hidden_dim", type=int, default=16)
parser.add_argument("--dropout", type=float, default=0.5)
parser.add_argument("--use-bias", type=bool, default=True)

# Training Parameters 
parser.add_argument("--lr", type=float, default=1e-2)
parser.add_argument("--weight_decay", type=float, default=5e-3)
parser.add_argument("--epochs", type=int, default=75)

config = parser.parse_args()
