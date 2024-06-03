import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.utils import k_hop_subgraph, to_dense_adj
import gudhi as gd
from torch_scatter import scatter
from torch_topological.utils import pairwise


def binary_gumbel_softmax(logits, tau=1.0, hard=False):
    """Samples from the Binary Gumbel-Softmax distribution."""
    gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits)))
    y = logits + gumbel_noise
    y = torch.sigmoid(y / tau)

    if hard:
        y_hard = (y > 0.5).float()
        y = y_hard - y.detach() + y

    return y


class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, num_heads=1):
        super(GraphAttentionLayer, self).__init__()
        self.num_heads = num_heads
        self.attention_heads = nn.ModuleList([
            nn.Linear(in_features, out_features) for _ in range(num_heads)
        ])
        self.attention_weights = nn.ParameterList([
            nn.Parameter(torch.Tensor(out_features, out_features)) for _ in range(num_heads)
        ])
        self.reset_parameters()

    def reset_parameters(self):
        for weight in self.attention_weights:
            nn.init.xavier_uniform_(weight.data, gain=1.414)

    def forward(self, x, edge_index):
        row, col = edge_index
        attention_scores = []

        for i in range(self.num_heads):
            head_x = self.attention_heads[i](x)
            attention_score = (head_x[row] @ self.attention_weights[i] @ head_x[col].t()).diag()
            attention_scores.append(attention_score)

        attention_scores = torch.stack(attention_scores, dim=1).mean(dim=1)
        attention_scores = torch.sigmoid(attention_scores)

        return attention_scores


class DeepSetLayer(nn.Module):
    """Simple equivariant deep set layer."""

    def __init__(self, dim_in, dim_out, aggregation_fn):
        super().__init__()
        self.Gamma = nn.Linear(dim_in, dim_out)
        self.Lambda = nn.Linear(dim_in, dim_out, bias=False)
        self.aggregation_fn = aggregation_fn

    def forward(self, x, batch):
        xm = scatter(x, batch, dim=0, reduce=self.aggregation_fn)
        xm = self.Lambda(xm)
        x = self.Gamma(x)
        x = x - xm[batch, :]
        return x


class TOGLWithSelfAttention(nn.Module):
    def __init__(self, n_features, n_filtrations, hidden_dim, out_dim, aggregation_fn, edge_threshold=100):
        super(TOGLWithSelfAttention, self).__init__()
        self.n_filtrations = n_filtrations
        self.filtrations = nn.Sequential(
            nn.Linear(n_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_filtrations),
        )
        self.set_fn = nn.ModuleList([
            nn.Linear(n_filtrations * 2, out_dim),
            nn.ReLU(),
            DeepSetLayer(out_dim, out_dim, aggregation_fn),
            nn.ReLU(),
            DeepSetLayer(out_dim, n_features, aggregation_fn),
        ])
        self.batch_norm = nn.BatchNorm1d(n_features)
        self.attention_layer = GraphAttentionLayer(n_features, 128, num_heads=4)
        self.edge_threshold = edge_threshold

    def compute_persistent_homology(self, x, edge_index, vertex_slices, edge_slices, batch, n_nodes):
        filtered_v = self.filtrations(x)
        filtered_e, _ = torch.max(
            torch.stack((filtered_v[edge_index[0]], filtered_v[edge_index[1]])),
            axis=0,
        )
        filtered_v = filtered_v.transpose(1, 0).to(x.device)
        filtered_e = filtered_e.transpose(1, 0).to(x.device)
        edge_index = edge_index.transpose(1, 0).to(x.device)
        vertex_index = torch.arange(end=n_nodes, dtype=torch.int).to(x.device)

        persistence_diagrams = torch.empty(
            (self.n_filtrations, n_nodes, 2),
            dtype=torch.float,
            device=x.device,
        )

        for filt_index in range(self.n_filtrations):
            for (vi, vj), (ei, ej) in zip(pairwise(vertex_slices), pairwise(edge_slices)):
                vertices = vertex_index[vi:vj]
                edges = edge_index[ei:ej]
                offset = vi
                f_vertices = filtered_v[filt_index][vi:vj]
                f_edges = filtered_e[filt_index][ei:ej]
                print(vi, vj)
                if len(vertices) == 0 or len(edges) == 0:
                    continue

                if len(edges) <= self.edge_threshold:
                    persistence_diagram = self._compute_persistent_homology(vertices, f_vertices, edges, f_edges,
                                                                            offset)
                else:
                    persistence_diagram = self._compute_0d_homology(vertices, f_vertices, offset)

                print(vi, vj)
                print(persistence_diagram.shape)
                persistence_diagrams[filt_index, vi:vj] = persistence_diagram

        return persistence_diagrams

    def _compute_persistent_homology(self, vertices, f_vertices, edges, f_edges, offset):
        st = gd.SimplexTree()

        for v, f in zip(vertices.cpu(), f_vertices.cpu()):
            st.insert([v.item()], filtration=f.item())
        for (u, v), f in zip(edges.cpu(), f_edges.cpu()):
            st.insert([u.item(), v.item()], filtration=f.item())
        st.make_filtration_non_decreasing()
        st.expansion(2)
        st.persistence()

        generators = st.lower_star_persistence_generators()
        generators_regular, _ = generators

        if len(generators_regular) == 0:
            return torch.stack((f_vertices, f_vertices), dim=1)

        generators_regular = torch.as_tensor(generators_regular[0], device=f_vertices.device)
        generators_regular = generators_regular - offset
        generators_regular = generators_regular.sort(dim=0, stable=True)[0]

        persistence_diagram = torch.stack((f_vertices, f_vertices), dim=1)
        if len(generators_regular) > 0:
            persistence_diagram[generators_regular[:, 0], 1] = f_vertices[generators_regular[:, 1]]

        return persistence_diagram

    def _compute_0d_homology(self, vertices, f_vertices, offset):
        st = gd.SimplexTree()

        for v, f in zip(vertices.cpu(), f_vertices.cpu()):
            st.insert([v.item()], filtration=f.item())
        st.make_filtration_non_decreasing()
        st.persistence()

        generators = st.lower_star_persistence_generators()
        generators_regular, _ = generators

        if len(generators_regular) == 0:
            return torch.stack((f_vertices, f_vertices), dim=1)

        generators_regular = torch.as_tensor(generators_regular[0], device=f_vertices.device)
        generators_regular = generators_regular - offset
        generators_regular = generators_regular.sort(dim=0, stable=True)[0]

        persistence_diagram = torch.stack((f_vertices, f_vertices), dim=1)
        if len(generators_regular) > 0:
            persistence_diagram[generators_regular[:, 0], 1] = f_vertices[generators_regular[:, 1]]

        return persistence_diagram

    def forward(self, x, data):
        edge_index = data.edge_index
        vertex_slices = torch.Tensor(data._slice_dict["x"]).long().to(x.device)
        edge_slices = torch.Tensor(data._slice_dict["edge_index"]).long().to(x.device)
        batch = data.batch.to(x.device)

        attention_weights = self.attention_layer(x, edge_index)
        #sampled_edges = torch.bernoulli(attention_weights).bool()
        sampled_edges = binary_gumbel_softmax(attention_weights, tau=1, hard=True)
        print
        topk_indices = torch.nonzero(sampled_edges).squeeze()
        edge_index = edge_index[:, topk_indices]

        count_indices = (topk_indices.unsqueeze(0) < edge_slices.unsqueeze(1)).sum(dim=1)
        edge_slices = edge_slices - count_indices

        persistence_pairs = self.compute_persistent_homology(
            x,
            edge_index,
            vertex_slices,
            edge_slices,
            batch,
            n_nodes=data.num_nodes,
        )

        x0 = persistence_pairs.permute(1, 0, 2).reshape(persistence_pairs.shape[1], -1)

        for layer in self.set_fn:
            if isinstance(layer, DeepSetLayer):
                x0 = layer(x0, batch)
            else:
                x0 = layer(x0)

        x = x + self.batch_norm(F.relu(x0))
        return x
