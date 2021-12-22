"""GUDHI integration demo."""

import numpy as np
import matplotlib.pyplot as plt

from torch_topological.nn import Cubical
from torch_topological.nn import SummaryStatisticLoss

from sklearn.datasets import make_circles

import torch


def _circle(i, j, n):
    r = np.sqrt((i - n/2.)**2 + (j - n/2.)**2)
    return np.exp(-(r - n/3.)**2/(n*2))


def _make_data(n_cells, n_samples=1000):
    X = make_circles(n_samples, shuffle=True, noise=0.05)[0]

    heatmap, *_ = np.histogram2d(X[:, 0], X[:, 1], bins=n_cells)
    heatmap -= heatmap.mean()
    heatmap /= heatmap.max()

    return heatmap


if __name__ == '__main__':

    np.random.seed(23)

    Y = _make_data(50)
    Y = torch.as_tensor(Y, dtype=torch.float)
    X = torch.as_tensor(
        Y + np.random.normal(scale=0.05, size=Y.shape), dtype=float
    )
    X = torch.nn.Parameter(X, requires_grad=True)

    optimizer = torch.optim.Adam([X], lr=1e-2)
    loss_fn = SummaryStatisticLoss('total_persistence', p=1)

    cubical = Cubical()

    persistence_information_target = cubical(Y)
    persistence_information_target = [persistence_information_target[0]]

    for i in range(500):
        persistence_information = cubical(X)
        persistence_information = [persistence_information[0]]

        optimizer.zero_grad()

        loss = loss_fn(
            persistence_information,
            persistence_information_target
        )

        print(loss.item())

        loss.backward()
        optimizer.step()

    X = X.detach().numpy()

    plt.imshow(X)
    plt.show()