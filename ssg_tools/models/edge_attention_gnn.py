#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Some codes here are modified from SuperGluePretrainedNetwork
# https://github.com/magicleap/SuperGluePretrainedNetwork/blob/master/models/superglue.py
#
# Code based on code from https://github.com/ShunChengWu/3DSSG/tree/cvpr21
# Copyright (c) 2021, ShunChengWu
# All rights reserved.


import torch
import torch.nn as nn
from .transformer.attention import MultiHeadAttention
from torch_geometric.nn.conv import MessagePassing
from .util import build_mlp
from typing import Optional


class Gen_Index(MessagePassing):
    """A sequence of scene graph convolution layers"""

    def __init__(self, flow="target_to_source"):
        super().__init__(flow=flow)

    def forward(self, x, edges_indices):
        size = self._check_input(edges_indices, None)
        coll_dict = self._collect(self._user_args, edges_indices, size, {"x": x})
        msg_kwargs = self.inspector.collect_param_data("message", coll_dict)
        x_i, x_j = self.message(**msg_kwargs)
        return x_i, x_j

    def message(self, x_i, x_j):
        return x_i, x_j


class Aggre_Index(MessagePassing):
    def __init__(self, aggr="add", node_dim=-2, flow="source_to_target"):
        super().__init__(aggr=aggr, node_dim=node_dim, flow=flow)

    def forward(self, x, edge_index, dim_size):
        size = self._check_input(edge_index, None)
        coll_dict = self._collect(self._user_args, edge_index, size, {})
        coll_dict["dim_size"] = dim_size
        aggr_kwargs = self.inspector.collect_param_data("aggregate", coll_dict)
        x = self.aggregate(x, **aggr_kwargs)
        return x


class TripletEdgeNet(torch.nn.Module):
    def __init__(self, dim_node, dim_edge, use_bn=False):
        super().__init__()
        self.name = "TripletEdgeNet"
        self.dim_node = dim_node
        self.dim_edge = dim_edge
        self.nn = build_mlp([dim_node * 2 + dim_edge, 2 * (dim_node + dim_edge), dim_edge], do_bn=use_bn, on_last=False)

    def forward(self, x_i, edge_feature, x_j):
        x_ = torch.cat([x_i, edge_feature, x_j], dim=1)  # .view(b, -1, 1)
        return self.nn(x_)


class MultiHeadedEdgeAttention(torch.nn.Module):
    def __init__(
        self,
        num_heads: int,
        dim_node: int,
        dim_edge: int,
        dim_atten: int,
        use_bn=False,
        attention="fat",
        use_edge: bool = True,
        drop_out_attention: Optional[float] = None,
    ):
        super().__init__()
        assert dim_node % num_heads == 0
        assert dim_edge % num_heads == 0
        assert dim_atten % num_heads == 0
        self.name = "MultiHeadedEdgeAttention"
        self.dim_node = dim_node
        self.dim_edge = dim_edge
        self.d_n = d_n = dim_node // num_heads
        self.d_e = d_e = dim_edge // num_heads
        self.d_o = d_o = dim_atten // num_heads
        self.num_heads = num_heads
        self.use_edge = use_edge
        self.nn_edge = build_mlp([dim_node * 2 + dim_edge, (dim_node + dim_edge), dim_edge], batch_norm=use_bn, on_last=False)

        self.attention = attention
        assert self.attention in ["fat"]

        if self.attention == "fat":
            if use_edge:
                self.nn = build_mlp([d_n + d_e, d_n + d_e, d_o], op="conv1d", batch_norm=use_bn, dropout=drop_out_attention)
            else:
                self.nn = build_mlp([d_n, d_n * 2, d_o], op="conv1d", batch_norm=use_bn, dropout=drop_out_attention)

            self.proj_edge = build_mlp([dim_edge, dim_edge])
            self.proj_query = build_mlp([dim_node, dim_node])
            self.proj_value = build_mlp([dim_node, dim_atten])
        else:
            raise NotImplementedError("")

    def forward(self, query, edge, value):
        batch_dim = query.size(0)
        edge_feature = self.nn_edge(torch.cat([query, edge, value], dim=1))  # .view(b, -1, 1)

        if self.attention == "fat":
            value = self.proj_value(value)
            query = self.proj_query(query).view(batch_dim, self.d_n, self.num_heads)
            edge = self.proj_edge(edge).view(batch_dim, self.d_e, self.num_heads)
            if self.use_edge:
                prob = self.nn(torch.cat([query, edge], dim=1))  # b, dim, head
            else:
                prob = self.nn(query)  # b, dim, head
            prob = prob.softmax(1)
            x = torch.einsum("bm,bm->bm", prob.reshape_as(value), value)

        return x, edge_feature, prob


class GraphEdgeAttenNetwork(nn.Module):
    def __init__(
        self,
        num_heads,
        dim_node,
        dim_edge,
        dim_atten,
        aggr="max",
        use_bn=False,
        flow="target_to_source",
        attention="fat",
        use_edge: bool = True,
        drop_out_attention: Optional[float] = None,
    ):
        super().__init__()  # "Max" aggregation.
        self.name = "edgeatten"
        self.dim_node = dim_node
        self.dim_edge = dim_edge
        self.index_get = Gen_Index(flow=flow)
        self.index_aggr = Aggre_Index(aggr=aggr, flow=flow)

        self.attention = attention
        assert self.attention in ["fat"]
        if self.attention == "fat":
            self.edgeatten = MultiHeadedEdgeAttention(
                dim_node=dim_node,
                dim_edge=dim_edge,
                dim_atten=dim_atten,
                num_heads=num_heads,
                use_bn=use_bn,
                attention=attention,
                use_edge=use_edge,
                drop_out_attention=drop_out_attention,
            )
            self.prop = build_mlp([dim_node + dim_atten, dim_node + dim_atten, dim_node], batch_norm=use_bn, on_last=False)
        else:
            raise NotImplementedError("")

    def forward(self, x, edge_feature, edge_index):
        assert x.ndim == 2
        assert edge_feature.ndim == 2
        x_i, x_j = self.index_get(x, edge_index)
        xx, gcn_edge_feature, prob = self.edgeatten(x_i, edge_feature, x_j)
        xx = self.index_aggr(xx, edge_index, dim_size=x.shape[0])
        xx = self.prop(torch.cat([x, xx], dim=1))
        return xx, gcn_edge_feature, prob


class GraphEdgeAttenNetworkLayers(torch.nn.Module):
    """A sequence of scene graph convolution layers"""

    def __init__(
        self,
        dim_node,
        dim_edge,
        dim_atten,
        num_layers,
        num_heads=1,
        aggr="max",
        use_bn=False,
        flow="target_to_source",
        attention="fat",
        use_edge: bool = True,
        drop_out_attention: Optional[float] = None,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.gconvs = torch.nn.ModuleList()

        self.drop_out = None
        if drop_out_attention is not None:
            self.drop_out = torch.nn.Dropout(drop_out_attention)

        self.self_attn = nn.ModuleList(
            MultiHeadAttention(d_model=dim_node, d_k=dim_node // num_heads, d_v=dim_node // num_heads, h=num_heads)
            for _ in range(num_layers)
        )

        self.self_attn_fc = nn.Sequential(  # 4 32 32 4(head)
            nn.Linear(4, 32),  # xyz, dist
            nn.ReLU(),
            nn.LayerNorm(32),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.LayerNorm(32),
            nn.Linear(32, 8),
        )

        for _ in range(self.num_layers):
            self.gconvs.append(
                GraphEdgeAttenNetwork(
                    num_heads,
                    dim_node,
                    dim_edge,
                    dim_atten,
                    aggr,
                    use_bn=use_bn,
                    flow=flow,
                    attention=attention,
                    use_edge=use_edge,
                    drop_out_attention=drop_out_attention,
                )
            )

    def forward(self, node_feature, edge_feature, edges_indices, obj_center, batch_ids):
        probs = list()

        if obj_center is not None:
            # get attention weight
            batch_size = batch_ids.max().item() + 1
            N_K = node_feature.shape[0]
            mask = torch.zeros(1, 1, N_K, N_K).to(device=node_feature.device)
            distance = torch.zeros(1, self.num_heads, N_K, N_K).to(device=node_feature.device)
            count = 0

            for i in range(batch_size):

                idx_i = torch.where(batch_ids == i)[0]
                mask[:, :, count : count + len(idx_i), count : count + len(idx_i)] = 1  # noqa E203

                center_A = obj_center[None, idx_i, :].clone().detach().repeat(len(idx_i), 1, 1)
                center_B = obj_center[idx_i, None, :].clone().detach().repeat(1, len(idx_i), 1)
                center_dist = center_A - center_B
                dist = center_dist.pow(2)
                dist = torch.sqrt(torch.sum(dist, dim=-1))[:, :, None]
                weights = torch.cat([center_dist, dist], dim=-1).unsqueeze(0)  # 1 N N 4

                dist_weights = self.self_attn_fc(weights).permute(0, 3, 1, 2)  # 1 num_heads N N

                attention_matrix_way = "add"
                distance[:, :, count : count + len(idx_i), count : count + len(idx_i)] = dist_weights  # noqa E203

                count += len(idx_i)
        else:
            mask = None
            distance = None
            attention_matrix_way = "mul"

        for i in range(self.num_layers):

            node_feature = node_feature.unsqueeze(0)
            node_feature = self.self_attn[i](
                node_feature,
                node_feature,
                node_feature,
                attention_weights=distance,
                way=attention_matrix_way,
                attention_mask=mask,
            )
            node_feature = node_feature.squeeze(0)

            gconv = self.gconvs[i]
            node_feature, edge_feature, prob = gconv(node_feature, edge_feature, edges_indices)

            if i < (self.num_layers - 1) or self.num_layers == 1:
                node_feature = torch.nn.functional.relu(node_feature)
                edge_feature = torch.nn.functional.relu(edge_feature)

                if self.drop_out:
                    node_feature = self.drop_out(node_feature)
                    edge_feature = self.drop_out(edge_feature)

            if prob is not None:
                probs.append(prob.cpu().detach())
            else:
                probs.append(None)
        return node_feature, edge_feature, probs
