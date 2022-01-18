from torch_topological.nn import CubicalComplex

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import numpy as np


class RandomDataSet(Dataset):
    def __init__(self, n_samples, dim, side_length, n_channels):
        self.dim = dim
        self.side_length = side_length
        self.n_samples = n_samples
        self.n_channels = n_channels

        self.data = np.random.default_rng().normal(
            size=(n_samples, n_channels, *([side_length] * dim))
        )

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


class TestCubicalComplex:
    batch_size = 32

    def test_2d(self):
        for n_channels in [1, 3]:
            for squeeze in [False, True]:
                data_set = RandomDataSet(128, 2, 8, n_channels)
                loader = DataLoader(
                    data_set,
                    self.batch_size,
                    shuffle=True,
                    drop_last=False
                )

                if squeeze:
                    data_set.data = data_set.data.squeeze()

                cc = CubicalComplex(dim=2)

                for batch in loader:
                    pers_info = cc(batch)

                    assert pers_info is not None

    def test_3d(self):
        data_set = RandomDataSet(128, 3, 8, 1)
        loader = DataLoader(
            data_set,
            self.batch_size,
            shuffle=True,
            drop_last=False
        )

        cc = CubicalComplex(dim=3)

        for batch in loader:
            pers_info = cc(batch)

            assert pers_info is not None
            assert len(pers_info) == self.batch_size