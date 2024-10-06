from argparse import ArgumentParser

parser = ArgumentParser()

# Model choice
parser.add_argument("--model", type=str, choices=["GAT", "GCN"], default="GAT")

# Dataset choice
parser.add_argument("--dataset", type=str, choices=["SQU", "CORA", "PPI"], default="PPI")

# For GAT 
parser.add_argument("--num_heads", type=int, nargs='+', default=[4,8,16]) # [4,8,16]

# Model Parameters (For both GAT & GCN)
parser.add_argument("--num_layers", type=int, nargs='+', default=[2]) # [2,3,4] - For the sake of less messy code, just manually change the varying variable here, 
parser.add_argument("--hidden_dim", type=int, nargs='+', default=[16]) #  [16,32,64] - Constant values (to ensure other variables are constant) are denoted with a list 
parser.add_argument("--dropout", type=float, nargs='+', default=[0.5])
parser.add_argument("--use-bias", type=bool, default=True)

# Training Parameters 
parser.add_argument("--lr", type=float, nargs='+', default=[0.001])
parser.add_argument("--weight_decay", type=float, default=5e-3)
parser.add_argument("--epochs", type=int, default=75)

config = parser.parse_args()