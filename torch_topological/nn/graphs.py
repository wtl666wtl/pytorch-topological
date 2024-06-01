"""Layers for topological data analysis based on graphs.

This is a work-in-progress module. The goal is to provide
a functionality similar to what is described in TOGL; see
https://github.com/BorgwardtLab/TOGL.

At the moment, the following functionality is present:
    - simple deep set layer
    - simple TOGL implementation with deep set functions
    - basic GCN with TOGL

The following aspects are currently ignored:
    - handling higher-order information properly
    - expanding simplicial complexes
    - making use of the dimension of features

On the other hand, the current implementation is simplified in the sense
of showing that *lower star filtrations* (and their corresponding
generators) may be useful.
"""

import torch
import torch.nn as nn

import gudhi as gd

from torch_geometric.data import Data

from torch_geometric.loader import DataLoader

from torch_geometric.utils import erdos_renyi_graph

from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool

from torch_scatter import scatter

from torch_topological.utils import pairwise


class DeepSetLayer(nn.Module):
    """Simple equivariant deep set layer."""

    def __init__(self, dim_in, dim_out, aggregation_fn):
        """Create new deep set layer.

        Parameters
        ----------
        dim_in : int
            Input dimension

        dim_out : int
            Output dimension

        aggregation_fn : str
            Aggregation to use for the reduction step. Must be valid for
            the ``torch_scatter.scatter()`` function, i.e. one of "sum",
            "mul", "mean", "min" or "max".
        """
        super().__init__()

        self.Gamma = nn.Linear(dim_in, dim_out)
        self.Lambda = nn.Linear(dim_in, dim_out, bias=False)

        self.aggregation_fn = aggregation_fn

    def forward(self, x, batch):
        """Implement forward pass through layer."""
        xm = scatter(x, batch, dim=0, reduce=self.aggregation_fn)
        xm = self.Lambda(xm)

        x = self.Gamma(x)
        x = x - xm[batch, :]
        return x


class TOGL(nn.Module):
    """Implementation of TOGL, a topological graph layer.

    Some caveats: this implementation only focuses on a set function
    aggregation of topological features. At the moment, it is not as
    powerful and feature-complete as the original implementation.
    """

    def __init__(
        self,
        n_features,
        n_filtrations,
        hidden_dim,
        out_dim,
        aggregation_fn,
    ):
        super().__init__()

        self.n_filtrations = n_filtrations

        self.filtrations = nn.Sequential(
            nn.Linear(n_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_filtrations),
        )

        self.set_fn = nn.ModuleList(
            [
                nn.Linear(n_filtrations * 2, out_dim),
                nn.ReLU(),
                DeepSetLayer(out_dim, out_dim, aggregation_fn),
                nn.ReLU(),
                DeepSetLayer(
                    out_dim,
                    n_features,
                    aggregation_fn,
                ),
            ]
        )

        self.batch_norm = nn.BatchNorm1d(n_features)

    def compute_persistent_homology(
        self,
        x,
        edge_index,
        vertex_slices,
        edge_slices,
        batch,
        n_nodes,
        return_filtration=False,
    ):
        """Return persistence pairs (i.e. generators)."""
        # Apply filtrations to node attributes. For the edge values, we
        # use a sublevel set filtration.
        #
        # TODO: Support different ways of filtering?
        filtered_v = self.filtrations(x)
        filtered_e, _ = torch.max(
            torch.stack(
                (filtered_v[edge_index[0]], filtered_v[edge_index[1]])
            ),
            axis=0,
        )

        filtered_v = filtered_v.transpose(1, 0).cpu().contiguous()
        filtered_e = filtered_e.transpose(1, 0).cpu().contiguous()
        edge_index = edge_index.cpu().transpose(1, 0).contiguous()

        # TODO: Do we have to enforce contiguous indices here?
        vertex_index = torch.arange(end=n_nodes, dtype=torch.int)

        # Fill all persistence information at the same time.
        persistence_diagrams = torch.empty(
            (self.n_filtrations, n_nodes, 2),
            dtype=torch.float,
        )

        for filt_index in range(self.n_filtrations):
            for (vi, vj), (ei, ej) in zip(
                pairwise(vertex_slices), pairwise(edge_slices)
            ):
                vertices = vertex_index[vi:vj]
                edges = edge_index[ei:ej]

                offset = vi

                f_vertices = filtered_v[filt_index][vi:vj]
                f_edges = filtered_e[filt_index][ei:ej]
                
                if len(vertices) == 0 or len(edges) == 0:
                        print(f"Skipping empty slice: vertices {len(vertices)}, edges {len(edges)}")
                        continue
                
                persistence_diagram = self._compute_persistent_homology(
                    vertices, f_vertices, edges, f_edges, offset
                )

                persistence_diagrams[filt_index, vi:vj] = persistence_diagram

        # Make sure that the tensor is living on the proper device here;
        # all subsequent operations can happen either on the CPU *or* on
        # the GPU.
        persistence_diagrams = persistence_diagrams.to(x.device)
        return persistence_diagrams

    # Helper function for doing the actual calculation of topological
    # features of a graph.
    def _compute_persistent_homology(
        self, vertices, f_vertices, edges, f_edges, offset
    ):
        assert len(vertices) == len(f_vertices)
        assert len(edges) == len(f_edges)

        st = gd.SimplexTree()

        for v, f in zip(vertices, f_vertices):
            st.insert([v], filtration=f)

        for (u, v), f in zip(edges, f_edges):
            st.insert([u, v], filtration=f)

        st.make_filtration_non_decreasing()
        st.expansion(2)
        st.persistence()

        # The generators are split into "regular" and "essential"
        # vertices, sorted by dimension.
        generators = st.lower_star_persistence_generators()
        generators_regular, generators_essential = generators
        
        if len(generators_regular) == 0:
            #print("Warning: generators_regular is empty")
            #print(vertices, f_vertices, edges, f_edges, offset)
            #print(st)
            return torch.stack((f_vertices, f_vertices), dim=1)

        # TODO: Let's think about how to leverage *all* generators in
        # *all* dimensions.
        generators_regular = torch.as_tensor(generators_regular[0])
        generators_regular = generators_regular - offset
        generators_regular = generators_regular.sort(dim=0, stable=True)[0]

        # By default, every vertex is paired with itself, so we just
        # duplicate the information here.
        persistence_diagram = torch.stack((f_vertices, f_vertices), dim=1)

        # Map generators back to filtration values, thus adding non-trivial
        # tuples to the persistence diagram while preserving gradients.
        if len(generators_regular) > 0:
            persistence_diagram[generators_regular[:, 0], 1] = f_vertices[
                generators_regular[:, 1]
            ]

        return persistence_diagram

    def forward(self, x, data):
        """Implement forward pass through data."""
        # TODO: Is this the best signature? `data` is following directly
        # the convention of `PyG`.
        #
        # x : current node attributes of layer; we should not use the
        # original attributes here because they are not informed by a
        # previous layer.
        #
        # data : edge slice information etc.

        edge_index = data.edge_index

        vertex_slices = torch.Tensor(data._slice_dict["x"]).long()
        edge_slices = torch.Tensor(data._slice_dict["edge_index"]).long()
        batch = data.batch

        persistence_pairs = self.compute_persistent_homology(
            x,
            edge_index,
            vertex_slices,
            edge_slices,
            batch,
            n_nodes=data.num_nodes,
        )

        x0 = persistence_pairs.permute(1, 0, 2).reshape(
            persistence_pairs.shape[1], -1
        )

        for layer in self.set_fn:
            # Preserve batch information for our set function layer
            # instead of treating all inputs the same.
            if isinstance(layer, DeepSetLayer):
                x0 = layer(x0, batch)
            else:
                x0 = layer(x0)

        # TODO: Residual step; could be made optional. Plus, the optimal
        # order of operations is not clear.
        x = x + self.batch_norm(nn.functional.relu(x0))
        return x


class TopoGCN(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.layers = nn.ModuleList([GCNConv(1, 32), GCNConv(32, 16)])

        self.pooling_fn = global_mean_pool
        self.togl = TOGL(32, 64, 64, 32, "mean")
        self.fc = nn.Linear(16, 10)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        for layer in self.layers[:1]:
            x = layer(x, edge_index)

        x = self.togl(x, data)
        
        for layer in self.layers[1:]:
            x = layer(x, edge_index)
        
        x = self.pooling_fn(x, data.batch)
        x = self.fc(x)
        
        return x


#B = 64
#N = 100
#p = 0.2

#if torch.cuda.is_available():
#    dev = "cuda:0"
#else:
#    dev = "cpu"

#print("Selected device:", dev)

#data_list = [
#    Data(x=torch.rand(N, 1), edge_index=erdos_renyi_graph(N, p), num_nodes=N)
#    for i in range(B)
#]

#loader = DataLoader(data_list, batch_size=8)

#model = TopoGCN().to(dev)

#for index, batch in enumerate(loader):
#    print(batch)
#    batch = batch.to(dev)

#    vertex_slices = torch.Tensor(batch._slice_dict["x"]).long()
#    edge_slices = torch.Tensor(batch._slice_dict["edge_index"]).long()

#    model(batch)
