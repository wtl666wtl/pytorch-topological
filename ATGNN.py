import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import GNNBenchmarkDataset
from tqdm import tqdm
import argparse
import random
import numpy as np
import time
from torch_topological.nn.graphs_new import TOGLWithSelfAttention


# GCN model with ATOGL layer
class TopoGCN(nn.Module):
    def __init__(self):
        super(TopoGCN, self).__init__()
        self.conv1 = GCNConv(1, 32)
        self.togl = TOGLWithSelfAttention(32, 8, 32, 32, "mean")
        self.conv3 = GCNConv(32, 128)
        self.conv4 = GCNConv(128, 256)
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.conv1(x, edge_index))
        x = self.togl(x, data)
        x = F.relu(self.conv3(x, edge_index))
        x = F.relu(self.conv4(x, edge_index))
        x = global_mean_pool(x, batch)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def train(model, train_loader, criterion, optimizer, device):
    model.train()
    correct = 0
    for data in tqdm(train_loader, desc="Training"):
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, data.y)
        loss.backward()
        optimizer.step()
        pred = output.max(1)[1]
        correct += pred.eq(data.y).sum().item()
    return correct / len(train_loader.dataset)


def test(model, loader, device):
    model.eval()
    correct = 0
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            output = model(data)
            pred = output.max(1)[1]
            correct += pred.eq(data.y).sum().item()
    return correct / len(loader.dataset)


def main():
    parser = argparse.ArgumentParser(description='ATGCN for GNN Benchmark Dataset')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train')
    parser.add_argument('--dataset', type=str, default='MNIST', help='Dataset name (e.g., MNIST)')
    parser.add_argument('--train_size', type=int, default=10000, help='Size of training dataset')
    parser.add_argument('--random_state', type=int, default=42, help='Random seed')

    args = parser.parse_args()

    # Set random seed
    set_seed(args.random_state)

    # Check for GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load dataset
    train_dataset = GNNBenchmarkDataset(root='data', name=args.dataset, split='train')
    test_dataset = GNNBenchmarkDataset(root='data', name=args.dataset, split='test')
    train_dataset = torch.utils.data.Subset(train_dataset, range(args.train_size))

    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # Initialize model, loss function, optimizer, and scheduler
    model = TopoGCN().to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {total_params}")
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=args.learning_rate)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Training loop
    start_time = time.time()
    for epoch in range(1, args.epochs + 1):
        train_acc = train(model, train_loader, criterion, optimizer, device)
        scheduler.step()
        print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}')
    total_time = time.time() - start_time
    print(f'Total training time: {total_time:.2f} seconds')

    test_acc = test(model, test_loader, device)
    print(f'Test Acc: {test_acc:.4f}')


if __name__ == '__main__':
    main()
