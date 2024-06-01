# main.py
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch_geometric.data import Data, DataLoader
from torch_geometric.utils import erdos_renyi_graph
from torch_topological.nn.graphs import TopoGCN
from tqdm import tqdm

transform = transforms.Compose([transforms.ToTensor()])
mnist_train = datasets.MNIST(root='data', train=True, download=True, transform=transform)
mnist_test = datasets.MNIST(root='data', train=False, download=True, transform=transform)

train_subset = torch.utils.data.Subset(mnist_train, range(100))
test_subset = torch.utils.data.Subset(mnist_test, range(100))


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
        loss = criterion(out, data.y)
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

    model = TopoGCN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(1, 21):
        train(model, train_loader, criterion, optimizer, device)
        train_acc = test(model, train_loader, device)
        test_acc = test(model, test_loader, device)
        print(f'Epoch: {epoch}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')


if __name__ == '__main__':
    main()
