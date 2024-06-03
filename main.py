import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import GNNBenchmarkDataset
from torch_topological.nn.graphs_old import TOGL
from tqdm import tqdm

# Parameters
batch_size = 1024
learning_rate = 0.01
epochs = 200

# Check for GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load dataset
train_dataset = GNNBenchmarkDataset(root='data', name='MNIST', split='train')
test_dataset = GNNBenchmarkDataset(root='data', name='MNIST', split='test')

train_dataset = torch.utils.data.Subset(train_dataset, range(10000))
test_dataset = torch.utils.data.Subset(test_dataset, range(1000))

# Data loaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# GCN model with TOGL layer
class TopoGCN(nn.Module):
    def __init__(self):
        super(TopoGCN, self).__init__()
        self.conv1 = GCNConv(1, 32)
        self.togl = TOGL(32, 8, 32, 32, "mean")
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

# Initialize model, loss function, optimizer, and scheduler
model = TopoGCN().to(device)
total_params = sum(p.numel() for p in model.parameters())
print(f"Total number of parameters: {total_params}")
criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=learning_rate)
scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

# Training and evaluation functions
def train():
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

def test(loader):
    model.eval()
    correct = 0
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            output = model(data)
            pred = output.max(1)[1]
            correct += pred.eq(data.y).sum().item()
    return correct / len(loader.dataset)

# Training loop
for epoch in range(1, epochs + 1):
    train_acc = train()
    test_acc = test(test_loader)
    scheduler.step()
    print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')
