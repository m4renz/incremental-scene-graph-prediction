from typing import Any, Dict, Iterable, Sequence

import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv, HANConv, HeteroConv, HeteroDictLinear, HGTConv, Linear, MessagePassing, SAGEConv


class EdgeFeatureEmbedder(torch.nn.Module):

    def __init__(
        self,
        input_dim: int = 11,
        hidden_dim: int = 64,
        output_dim: int = 64,
        dropout: float = 0.5,
        norm: str | None = None,
        edge_types: Sequence[tuple[str, str, str]] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        super().__init__()
        # TODO: Hetero?

        self.lin1 = Linear(input_dim, hidden_dim)
        self.lin2 = Linear(hidden_dim, output_dim)
        self.dropout = torch.nn.Dropout(dropout)
        self.edge_types = edge_types
        self.device = device

        self.norm = norm
        if self.norm == "layer":
            self.norm1 = torch.nn.LayerNorm(hidden_dim, elementwise_affine=True)
        elif self.norm == "batch":
            self.norm1 = torch.nn.BatchNorm1d(hidden_dim)

    def forward(self, descriptor_dict: Dict[Any, torch.Tensor], edge_index_dict: Dict[Any, torch.Tensor]):
        """
        descriptor_dict: Dictionary produced by HeteroData.x_dict / HeteroData.descriptor_dict.
            Needs to contain the 11 descriptor values for each node.
        edge_index_dict: Dictionary produced by HeteroData.edge_index_dict
        """

        edge_feature_dict = {}

        if self.edge_types:
            keys = [edge for edge in edge_index_dict.keys() if edge in self.edge_types]
        else:
            keys = edge_index_dict.keys()

        for edge in keys:
            x_target = descriptor_dict[edge[0]]  # wrong notation here, edge[0] is source, edge[2] is target
            x_source = descriptor_dict[edge[2]]
            edge_index = edge_index_dict[edge]
            edge_feature = self.get_features(
                x_target, x_source, edge_index
            )  # but same switch in get_features leads to correct source_to_target

            if not self.norm:
                edge_feature = F.relu(self.lin1(edge_feature))
                edge_feature = self.dropout(edge_feature)
                edge_feature = self.lin2(edge_feature)
            else:
                edge_feature = self.norm1(F.relu(self.lin1(edge_feature)))
                edge_feature = self.dropout(edge_feature)
                edge_feature = self.lin2(edge_feature)

            edge_feature_dict[edge] = edge_feature

        return edge_feature_dict

    def get_features(self, x_target, x_source, edge_index):
        # source_to_target
        # (j, i)
        # switch of source and target in upper function leads to edge_index[0] == source and edge_index[1] == target as it should be
        # means notation of x_i and x_j is technically switched but still correct with source_to_target
        # 0-2: centroid, 3-5: std, 6-8:dims, 9:volume, 10:length
        # to
        # 0-2: offset centroid, 3-5: offset std, 6-8: dim log ratio, 9: volume log ratio, 10: length log ratio
        x_i = x_target[edge_index[0]]
        x_j = x_source[edge_index[1]]

        # Compute edge features in a single operation
        edge_feature = torch.cat(
            [
                x_i[:, 0:3] - x_j[:, 0:3],  # centroid offset
                x_i[:, 3:6] - x_j[:, 3:6],  # std offset
                torch.log(x_i[:, 6:9] / x_j[:, 6:9]),  # extent log ratio
                torch.log(x_i[:, 9:10] / x_j[:, 9:10]),  # max_l log ratio
                torch.log(x_i[:, 10:11] / x_j[:, 10:11]),  # volume log ratio
            ],
            dim=1,
        )

        return edge_feature.to(x_target)  # same device and type


class EdgeFeatureEmbedderHomo(torch.nn.Module):

    def __init__(
        self,
        input_dim: int = 11,
        hidden_dim: int = 64,
        output_dim: int = 64,
        dropout: float = 0.5,
        norm: str | None = None,
    ):
        super().__init__()
        # TODO: Hetero?

        self.lin1 = Linear(input_dim, hidden_dim)
        self.lin2 = Linear(hidden_dim, output_dim)
        self.dropout = torch.nn.Dropout(dropout)
        # TODO: Fix this!

        self.norm = norm
        if self.norm == "layer":
            self.norm1 = torch.nn.LayerNorm(hidden_dim, elementwise_affine=True)
        elif self.norm == "batch":
            self.norm1 = torch.nn.BatchNorm1d(hidden_dim)

    def forward(self, descriptor: torch.Tensor, edge_index: torch.Tensor):
        """
        descriptor: 11 descriptor values for each node (num_nodes, 11).
        edge_index: Edge index tensor of shape (2, num_edges)
        """

        edge_feature = self.get_features(descriptor, edge_index)

        if not self.norm:
            edge_feature = F.relu(self.lin1(edge_feature))
            edge_feature = self.dropout(edge_feature)
            edge_feature = self.lin2(edge_feature)
        else:
            edge_feature = self.norm1(F.relu(self.lin1(edge_feature)))
            edge_feature = self.dropout(edge_feature)
            edge_feature = self.lin2(edge_feature)

        return edge_feature

    def get_features(self, descriptor, edge_index):
        # source_to_target
        # (j, i)
        # 0-2: centroid, 3-5: std, 6-8:dims, 9:volume, 10:length
        # to
        # 0-2: offset centroid, 3-5: offset std, 6-8: dim log ratio, 9: volume log ratio, 10: length log ratio
        x_i = descriptor[edge_index[0]]
        x_j = descriptor[edge_index[1]]

        # Compute edge features in a single operation
        edge_feature = torch.cat(
            [
                x_i[:, 0:3] - x_j[:, 0:3],  # centroid offset
                x_i[:, 3:6] - x_j[:, 3:6],  # std offset
                torch.log(x_i[:, 6:9] / x_j[:, 6:9]),  # dim log ratio
                torch.log(x_i[:, 9:10] / x_j[:, 9:10]),  # volume log ratio
                torch.log(x_i[:, 10:11] / x_j[:, 10:11]),  # length log ratio
            ],
            dim=1,
        )

        return edge_feature.to(descriptor)  # same device and type


class SceneGAT(torch.nn.Module):
    def __init__(
        self,
        node_channels: int | tuple[int, int],
        edge_channels: int,
        hidden_channels: int,
        out_channels: int,
        heads: int,
        num_layers: int,
        dropout: float,
        norm: str | None = None,
    ):
        super(SceneGAT, self).__init__()
        self.name = "SceneGAT"
        self.convs = torch.nn.ModuleList()
        self.num_layers = num_layers
        self.convs.append(
            GATConv(
                node_channels,
                hidden_channels,
                edge_dim=edge_channels,
                heads=heads,
                dropout=dropout,
                add_self_loops=False,
            )
        )
        for _ in range(num_layers - 2):
            self.convs.append(
                GATConv(
                    hidden_channels * heads,
                    hidden_channels,
                    edge_dim=edge_channels,
                    heads=heads,
                    dropout=dropout,
                    add_self_loops=False,
                )
            )
        self.convs.append(
            GATConv(
                hidden_channels * heads,
                out_channels,
                edge_dim=edge_channels,
                heads=1,
                concat=False,
                dropout=dropout,
                add_self_loops=False,
            )
        )

        self.norm = norm
        if self.norm == "layer":
            self.norms = torch.nn.ModuleList()
            for _ in range(num_layers - 1):
                self.norms.append(torch.nn.LayerNorm(hidden_channels * heads, elementwise_affine=True))
        elif self.norm == "batch":
            self.norms = torch.nn.ModuleList()
            for _ in range(num_layers - 1):
                self.norms.append(torch.nn.BatchNorm1d(hidden_channels * heads))

    def forward(self, x, edge_index, edge_features):
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index, edge_attr=edge_features)
            if i < self.num_layers - 1:
                x = F.relu(x)
                if self.norm:
                    x = self.norms[i](x)
        return x


class SceneSAGE(torch.nn.Module):
    def __init__(
        self,
        node_channels: int | tuple[int, int],
        hidden_channels: int,
        out_channels: int,
        num_layers: int,
        dropout: float,
        norm: str | None = None,
    ):
        super(SceneSAGE, self).__init__()
        self.name = "SceneSAGE"
        self.convs = torch.nn.ModuleList()
        self.num_layers = num_layers
        self.dropout = torch.nn.Dropout(dropout)

        self.convs.append(SAGEConv(node_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))

        self.norm = norm
        if self.norm == "layer":
            self.norms = torch.nn.ModuleList()
            for _ in range(num_layers - 1):
                self.norms.append(torch.nn.LayerNorm(hidden_channels, elementwise_affine=True))
        elif self.norm == "batch":
            self.norms = torch.nn.ModuleList()
            for _ in range(num_layers - 1):
                self.norms.append(torch.nn.BatchNorm1d(hidden_channels))

    def forward(self, x, edge_index, **kwargs):
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            if i < self.num_layers - 1:
                x = F.relu(x)
                if self.norm:
                    x = self.norms[i](x)
                x = self.dropout(x)
        return x


class SceneHGT(torch.nn.Module):
    def __init__(
        self,
        node_channels: int | tuple[int, int],
        hidden_channels: int,
        out_channels: int,
        heads: int,
        num_layers: int,
        dropout: float,
        metadata: tuple,
        norm: str | None = None,
    ):
        super(SceneHGT, self).__init__()
        self.name = "SceneHGT"
        self.convs = torch.nn.ModuleList()
        self.num_layers = num_layers
        self.dropout = torch.nn.Dropout(dropout)
        self.metadata = metadata

        self.convs.append(
            HGTConv(
                node_channels,
                hidden_channels,
                heads=heads,
                metadata=metadata,
            )
        )

        for _ in range(num_layers - 2):
            self.convs.append(
                HGTConv(
                    hidden_channels,
                    hidden_channels,
                    heads=heads,
                    metadata=metadata,
                )
            )
        self.convs.append(
            HGTConv(
                hidden_channels,
                out_channels=out_channels,
                heads=heads,
                metadata=metadata,
            )
        )

        self.norm = norm
        if self.norm == "layer":
            self.norms = torch.nn.ModuleList()
            for _ in range(num_layers - 1):
                self.norms.append(torch.nn.LayerNorm(hidden_channels, elementwise_affine=True))
        elif self.norm == "batch":
            self.norms = torch.nn.ModuleList()
            for _ in range(num_layers - 1):
                self.norms.append(torch.nn.BatchNorm1d(hidden_channels))

    def forward(self, x_dict, edge_index_dict, **kwargs):
        for i in range(self.num_layers):
            x_dict = self.convs[i](x_dict, edge_index_dict)
            if i < self.num_layers - 1:
                x_dict = {k: F.relu(v) for k, v in x_dict.items()}
                if self.norm:
                    x_dict = {k: self.norms[i](v) for k, v in x_dict.items()}
                x_dict = {k: self.dropout(v) for k, v in x_dict.items()}
        return x_dict


class SceneHAN(torch.nn.Module):
    def __init__(
        self,
        node_channels: int,
        hidden_channels: int,
        out_channels: int,
        heads: int,
        num_layers: int,
        dropout: float,
        attn_dropout: float,
        metadata: tuple,
        norm: str | None = None,
    ):
        super(SceneHAN, self).__init__()
        self.name = "SceneHAN"
        self.convs = torch.nn.ModuleList()
        self.num_layers = num_layers
        self.dropout = torch.nn.Dropout(dropout)
        self.metadata = metadata

        self.convs.append(
            HANConv(
                node_channels,
                hidden_channels,
                heads=heads,
                metadata=metadata,
                dropout=attn_dropout,
            )
        )

        for _ in range(num_layers - 2):
            self.convs.append(
                HANConv(
                    hidden_channels,
                    hidden_channels,
                    heads=heads,
                    metadata=metadata,
                    dropout=attn_dropout,
                )
            )

        self.convs.append(
            HANConv(
                hidden_channels,
                out_channels=out_channels,
                heads=heads,
                metadata=metadata,
                dropout=attn_dropout,
            )
        )

        self.norm = norm
        if self.norm == "layer":
            self.norms = torch.nn.ModuleList()
            for _ in range(num_layers - 1):
                self.norms.append(torch.nn.LayerNorm(hidden_channels, elementwise_affine=True))
        elif self.norm == "batch":
            self.norms = torch.nn.ModuleList()
            for _ in range(num_layers - 1):
                self.norms.append(torch.nn.BatchNorm1d(hidden_channels))

    def forward(self, x_dict, edge_index_dict, **kwargs):
        for i in range(self.num_layers):
            x_dict = self.convs[i](x_dict, edge_index_dict)
            if i < self.num_layers - 1:
                x_dict = {k: F.relu(v) for k, v in x_dict.items()}
                if self.norm:
                    x_dict = {k: self.norms[i](v) for k, v in x_dict.items()}
                x_dict = {k: self.dropout(v) for k, v in x_dict.items()}
        return x_dict


class EdgeMLP(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, types: Sequence[Any], dropout: float = 0.5, norm: str | None = None):

        super(EdgeMLP, self).__init__()
        self.types = [str(t) for t in types]
        self.types_map = {str(t): t for t in types}
        self.linear = HeteroDictLinear(in_channels, out_channels, types=self.types)
        self.dropout = torch.nn.Dropout(dropout)

        self.norm = norm
        if self.norm == "layer":
            self.norm = torch.nn.LayerNorm(out_channels, elementwise_affine=True)
        elif self.norm == "batch":
            self.norm = torch.nn.BatchNorm1d(out_channels)

    def forward(
        self, edge_features: Dict[str, torch.Tensor], node_features: Dict[str, torch.Tensor], edge_index_dict: Dict[str, torch.Tensor]
    ):

        out = {}
        _out = {}

        for e, i in edge_index_dict.items():
            if str(e) not in self.types:
                continue
            x_i = node_features[e[0]][i[0]]
            x_j = node_features[e[2]][i[1]]
            _out[str(e)] = self.dropout(torch.cat([x_i, edge_features[e], x_j], dim=1))

        _out = self.linear(_out)
        for k, v in _out.items():
            if self.norm:
                v = self.norm(v)
            out[self.types_map[k]] = v
        return out


class LinearClassifier(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.5, norm: str | None = None):
        super(LinearClassifier, self).__init__()
        self.lin1 = torch.nn.Linear(in_channels, in_channels)
        self.lin2 = torch.nn.Linear(in_channels, out_channels)
        self.dropout = torch.nn.Dropout(dropout)

        self.norm = norm
        if self.norm == "layer":
            self.norm = torch.nn.LayerNorm(in_channels, elementwise_affine=True)
        elif self.norm == "batch":
            self.norm = torch.nn.BatchNorm1d(in_channels)

    def forward(self, x):
        x = self.lin1(x)
        x = F.relu(x)
        if self.norm:
            x = self.norm(x)
        x = self.dropout(x)
        x = self.lin2(x)
        return x


class HeteroClassifier(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, types: Sequence[Any], dropout: float = 0.5, norm: str | None = None):
        super(HeteroClassifier, self).__init__()
        self.types = [str(t) for t in types]
        self.lin1 = HeteroDictLinear(in_channels, in_channels, types=self.types)
        self.lin2 = HeteroDictLinear(in_channels, out_channels, types=self.types)
        self.dropout = torch.nn.Dropout(dropout)

        self.norm = norm
        if self.norm == "layer":
            self.norm = torch.nn.LayerNorm(in_channels, elementwise_affine=True)
        elif self.norm == "batch":
            self.norm = torch.nn.BatchNorm1d(in_channels)

    def forward(self, x_dict):
        x = {k: v for k, v in x_dict.items() if k in self.types}
        x = self.lin1(x)
        x = {k: F.relu(v) for k, v in x.items()}
        if self.norm:
            x = {k: self.norm(v) for k, v in x.items()}
        x = {k: self.dropout(v) for k, v in x.items()}
        x = self.lin2(x)
        return x


class Classifier(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        multilabel: bool = False,
        dropout: float = 0.5,
        norm: str | None = None,
    ):
        super(Classifier, self).__init__()

        self.multilabel = multilabel

        self.lin1 = Linear(in_channels, in_channels)
        self.lin2 = Linear(in_channels, out_channels)
        self.dropout = torch.nn.Dropout(dropout)

        self.norm = norm
        if self.norm == "layer":
            self.norm = torch.nn.LayerNorm(in_channels, elementwise_affine=True)
        elif self.norm == "batch":
            self.norm = torch.nn.BatchNorm1d(in_channels)

    def forward(self, x):
        x = self.lin1(x)
        x = F.relu(x)
        if self.norm:
            x = self.norm(x)
        x = self.dropout(x)
        x = self.lin2(x)
        if self.multilabel:
            return x
        else:
            x = F.log_softmax(x, dim=-1)
        return x


class EdgeConv(MessagePassing):
    def __init__(self, in_channels: int, out_channels, edge_channels=-1, aggr="sum"):

        super(EdgeConv, self).__init__(aggr=aggr)

        if isinstance(in_channels, tuple):
            in_channels = in_channels[0]

        self.lin_edge = Linear(edge_channels, out_channels)
        self.lin_node = Linear(in_channels, out_channels)

    def forward(self, x, edge_index, edge_features):

        if edge_index.size(1) == 0:
            return self.lin_node(x)
        out = self.propagate(edge_index, x=x, edge_features=edge_features, size=(x.size(0), x.size(0)))
        return out

    def message(self, edge_features):
        return self.lin_edge(edge_features)

    def update(self, aggr_out, x):
        return super().update(self.lin_node(x) + aggr_out)


class HierarchicalHeteroConv(torch.nn.Module):

    def __init__(
        self,
        edge_types: Iterable[tuple[str, str, str]],
        node_channels: tuple[int, int],
        hidden_channels: int,
        out_channels: int,
        edge_channels: int,
        num_layers: int,
        dropout: float,
        norm: str | None = None,
    ):
        super(HierarchicalHeteroConv, self).__init__()
        self.name = "HierarchicalHeteroConv"
        self.convs = torch.nn.ModuleList()
        self.num_layers = num_layers
        self.dropout = torch.nn.Dropout(dropout)

        _edge_type_layers = {
            ("new", "to", "new"): SAGEConv,
            ("old", "to", "new"): SAGEConv,
            ("old", "to", "old"): SAGEConv,
            ("old", "h", "old"): EdgeConv,
        }

        self.convs.append(HeteroConv({k: _edge_type_layers[k](node_channels, hidden_channels) for k in edge_types}))

        for _ in range(num_layers - 2):
            self.convs.append(HeteroConv({k: _edge_type_layers[k](hidden_channels, hidden_channels) for k in edge_types}))

        self.convs.append(HeteroConv({k: _edge_type_layers[k](hidden_channels, out_channels) for k in edge_types}))

        self.norm = norm
        if self.norm == "layer":
            self.norms = torch.nn.ModuleList()
            for _ in range(num_layers - 1):
                self.norms.append(torch.nn.LayerNorm(hidden_channels, elementwise_affine=True))
        elif self.norm == "batch":
            self.norms = torch.nn.ModuleList()
            for _ in range(num_layers - 1):
                self.norms.append(torch.nn.BatchNorm1d(hidden_channels))

    def forward(self, x_dict, edge_index_dict, edge_features):
        for i in range(self.num_layers):
            x_dict = self.convs[i](x_dict, edge_index_dict, edge_features_dict=edge_features)
            if i < self.num_layers - 1:
                x_dict = {k: F.relu(v) for k, v in x_dict.items()}
                if self.norm:
                    x_dict = {k: self.norms[i](v) for k, v in x_dict.items()}
                x_dict = {k: self.dropout(v) for k, v in x_dict.items()}
        return x_dict
