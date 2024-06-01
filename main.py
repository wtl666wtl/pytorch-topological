# main.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.nn import global_mean_pool, GCNConv
from torchvision import datasets, transforms
from torch_geometric.data import Data, DataLoader
from torch_geometric.utils import erdos_renyi_graph
from torch_topological.nn.graphs import TopoGCN
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup
import matplotlib.pyplot as plt

transform = transforms.Compose([transforms.ToTensor()])
mnist_train = datasets.MNIST(root='data', train=True, download=True, transform=transform)
mnist_test = datasets.MNIST(root='data', train=False, download=True, transform=transform)

train_subset = torch.utils.data.Subset(mnist_train, range(10000))
test_subset = torch.utils.data.Subset(mnist_test, range(1000))


class GCN(torch.nn.Module):
    def __init__(self):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(1, 128)
        self.conv2 = GCNConv(128, 64)
        self.conv3 = GCNConv(64, 32)
        self.pool = global_mean_pool
        self.fc = nn.Linear(32, 10)
        self.dropout = nn.Dropout(0.5)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.conv3(x, edge_index)
        x = torch.relu(x)
        x = self.pool(x, data.batch)
        x = self.fc(x)
        return x


def convert_to_graph(data):
    data_list = []
    for img, label in data:
        img = img.view(28, 28).numpy()
        nodes = []
        edges = []
        for i in range(28):
            for j in range(28):
                nodes.append([img[i, j]])
                if i > 0:
                    edges.append([i * 28 + j, (i - 1) * 28 + j])
                    edges.append([(i - 1) * 28 + j, i * 28 + j])
                if j > 0:
                    edges.append([i * 28 + j, i * 28 + (j - 1)])
                    edges.append([i * 28 + (j - 1), i * 28 + j])
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        x = torch.tensor(nodes, dtype=torch.float)
        y = torch.tensor(label, dtype=torch.long).unsqueeze(0)
        data_list.append(Data(x=x, edge_index=edge_index, y=y, num_nodes=len(nodes)))
    return data_list


train_data = convert_to_graph(train_subset)
test_data = convert_to_graph(test_subset)

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)


def train(model, loader, criterion, optimizer, device):
    model.train()
    for data in tqdm(loader, desc="Training"):
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out, data.y.view(-1))
        loss.backward()
        optimizer.step()


@torch.no_grad()
def test(model, loader, device):
    model.eval()
    correct = 0
    for data in loader:
        data = data.to(device)
        out = model(data)
        pred = out.argmax(dim=1)
        correct += pred.eq(data.y).sum().item()
    return correct / len(loader.dataset)


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Selected device:", device)

    model = GCN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0005)

    # 使用 ReduceLROnPlateau 调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, verbose=True)

    num_epochs = 50
    best_test_acc = 0
    train_accuracies = []
    test_accuracies = []

    for epoch in range(1, num_epochs + 1):
        train(model, train_loader, criterion, optimizer, device)
        train_acc = test(model, train_loader, device)
        test_acc = test(model, test_loader, device)
        train_accuracies.append(train_acc)
        test_accuracies.append(test_acc)
        print(f'Epoch: {epoch}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')

        scheduler.step(test_acc)

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            if epoch > 10 and test_acc <= test_accuracies[-10]:
                print(f"Early stopping at epoch {epoch}")
                break

    plt.figure()
    plt.plot(range(1, len(train_accuracies) + 1), train_accuracies, label='Train Accuracy')
    plt.plot(range(1, len(test_accuracies) + 1), test_accuracies, label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('accuracy_curve.png')


if __name__ == '__main__':
    main()
