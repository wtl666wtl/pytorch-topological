import torch
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from torch_geometric.data import Data
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch.optim import Adam
from torch_geometric.loader import DataLoader
from tqdm import tqdm


batch_size = 64
learning_rate = 0.01
epochs = 20

transform = transforms.Compose([transforms.ToTensor()])
mnist_train = MNIST(root='data', train=True, download=True, transform=transform)
mnist_test = MNIST(root='data', train=False, download=True, transform=transform)

mnist_train = torch.utils.data.Subset(mnist_train, range(10000))
mnist_test = torch.utils.data.Subset(mnist_test, range(1000))


def mnist_to_graph(data):
    graphs = []
    for img, label in data:
        x = img.view(28 * 28, 1)
        edge_index = create_edge_index(28, 28)
        graph = Data(x=x, edge_index=edge_index, y=label)
        graphs.append(graph)
    return graphs

def create_edge_index(height, width):
    edge_index = []
    for i in range(height):
        for j in range(width):
            node = i * width + j
            if i > 0:
                edge_index.append([node, node - width])
            if i < height - 1:
                edge_index.append([node, node + width])
            if j > 0:
                edge_index.append([node, node - 1])
            if j < width - 1:
                edge_index.append([node, node + 1])
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    return edge_index


train_graphs = mnist_to_graph(mnist_train)
test_graphs = mnist_to_graph(mnist_test)


class GCN(nn.Module):
    def __init__(self):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(1, 32)
        self.conv2 = GCNConv(32, 64)
        self.fc1 = nn.Linear(64 * 28 * 28, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = x.view(-1, 64 * 28 * 28)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x



train_loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_graphs, batch_size=batch_size, shuffle=False)

model = GCN()
criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=learning_rate)


def train():
    model.train()
    for data in tqdm(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, data.y)
        loss.backward()
        optimizer.step()


@torch.no_grad()
def test(loader):
    model.eval()
    correct = 0
    for data in loader:
        output = model(data)
        pred = output.max(1)[1]
        correct += pred.eq(data.y).sum().item()
    return correct / len(loader.dataset)


for epoch in range(1, epochs + 1):
    train()
    train_acc = test(train_loader)
    test_acc = test(test_loader)
    print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')
