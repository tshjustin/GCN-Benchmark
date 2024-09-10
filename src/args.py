from argparse import ArgumentParser

parser = ArgumentParser()

parser.add_argument("--nodes_path", type=str, default="cora/cora.content")
parser.add_argument("--edges_path", type=str, default="cora/cora.cites")

parser.add_argument("--hidden_dim", type=int, default=16)
parser.add_argument("--dropout", type=float, default=0.5)
parser.add_argument("--use-bias", type=bool, default=True)

parser.add_argument("--train_size_per_class", type=int, default=20)
parser.add_argument("--validation_size", type=int, default=500)
parser.add_argument("--test_size", type=int, default=1000)

parser.add_argument("--lr", type=float, default=1e-2)
parser.add_argument("--weight_decay", type=float, default=5e-3)
parser.add_argument("--patience", type=int, default=10)
parser.add_argument("--epochs", type=int, default=200)
parser.add_argument("--use_early_stopping", type=bool, default=True)

config = parser.parse_args()
