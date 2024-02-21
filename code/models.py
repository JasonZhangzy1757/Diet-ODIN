import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv, GCNConv, GATConv, HANConv
from torch_sparse import SparseTensor
from utils import *
from torch_geometric.nn.conv.gcn_conv import gcn_norm
import torch

from torch_geometric.nn import HeteroConv, SAGEConv, global_mean_pool
from torch_geometric.nn import HGTConv, Linear
import torch_geometric.transforms as T

from torch_geometric.utils.hetero import construct_bipartite_edge_index
import math
from torch_geometric.utils import softmax


class AttHGT(torch.nn.Module):
    def __init__(self, graph, refined_graph, hidden_dim, num_classes, num_heads=4, num_layers=1, dropout=0.6):
        super().__init__()
        self.lin_dict = torch.nn.ModuleDict()
        for node_type in graph.node_types:
            self.lin_dict[node_type] = Linear(-1, hidden_dim)
        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = HGTConv_withAtt(hidden_dim, hidden_dim, graph.metadata(),
                                   num_heads, group='sum')
            self.convs.append(conv)
        self.han_conv = HANConv(-1, 64, heads=num_heads,
                                dropout=0.6, metadata=refined_graph.metadata())
        self.lin = nn.Linear(hidden_dim + 64, hidden_dim)

        self.lin1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, num_classes)
        self.dropout = dropout
        
        # Ablation_without_c
        # self.direct = nn.Linear(hidden_dim, num_classes)

    def forward(self, x_dict, edge_index_dict, x_dict_refined, edge_index_dict_refined):
        x_dict_refined = self.han_conv(x_dict_refined, edge_index_dict_refined)
        x_dict = {
            node_type: self.lin_dict[node_type](x).relu_()
            for node_type, x in x_dict.items()
        }
        attn_score_list = []
        for conv in self.convs:
            x_dict, attn_score = conv(x_dict, edge_index_dict)
            attn_score_list.append(attn_score)

        x_embedding = torch.cat([x_dict['user'], x_dict_refined['user']], dim=1)
        x_embedding = self.lin(x_embedding)
        
        # Ablation: Without a
        # x_embedding = x_dict['user']
        
        # Ablation: without c
        # x_embedding = self.direct(x_embedding)
        

        return x_embedding, attn_score_list

    def edge_pair_forward(self, edge_index, x):
        src, dst = edge_index
        src = x[src]
        dst = x[dst]
        edge_embedding = torch.cat([src, dst], dim=1)

        edge_embedding = self.lin1(edge_embedding)
        edge_embedding = F.relu(edge_embedding)
        edge_embedding = F.dropout(edge_embedding, training=self.training, p=self.dropout)
        edge_embedding = self.lin2(edge_embedding)

        return F.log_softmax(edge_embedding, dim=1)


class HGTConv_withAtt(HGTConv):
    def __init__(self, *args, **kwargs):
        super(HGTConv_withAtt, self).__init__(*args, **kwargs)
        # Additional initialization if needed

    def forward(self, x_dict, edge_index_dict):

        F = self.out_channels
        H = self.heads
        D = F // H

        k_dict, q_dict, v_dict, out_dict = {}, {}, {}, {}

        kqv_dict = self.kqv_lin(x_dict)
        for key, val in kqv_dict.items():
            k, q, v = torch.tensor_split(val, 3, dim=1)
            k_dict[key] = k.view(-1, H, D)
            q_dict[key] = q.view(-1, H, D)
            v_dict[key] = v.view(-1, H, D)

        q, dst_offset = self._cat(q_dict)
        k, v, src_offset = self._construct_src_node_feat(
            k_dict, v_dict, edge_index_dict)

        edge_index, edge_attr = construct_bipartite_edge_index(
            edge_index_dict, src_offset, dst_offset, edge_attr_dict=self.p_rel)

        out = self.propagate(edge_index, k=k, q=q, v=v, edge_attr=edge_attr,
                             size=None, return_attention_weights=True)
        attn_scores = self._attn_scores

        for node_type, start_offset in dst_offset.items():
            end_offset = start_offset + q_dict[node_type].size(0)
            if node_type in self.dst_node_types:
                out_dict[node_type] = out[start_offset:end_offset]

        a_dict = self.out_lin({
            k: torch.nn.functional.gelu(v) if v is not None else v
            for k, v in out_dict.items()
        })

        for node_type, out in out_dict.items():
            out = a_dict[node_type]
            if out.size(-1) == x_dict[node_type].size(-1):
                alpha = self.skip[node_type].sigmoid()
                out = alpha * out + (1 - alpha) * x_dict[node_type]
            out_dict[node_type] = out

        return out_dict, attn_scores

    def message(self, edge_index, k_j, q_i, v_j, edge_attr, ptr, size_i, return_attention_weights=True):
        # Override the message method to calculate and store attention weights
        alpha = (q_i * k_j).sum(dim=-1) * edge_attr
        alpha = alpha / math.sqrt(q_i.size(-1))
        alpha = softmax(alpha, edge_index[1], ptr, size_i)
        if return_attention_weights:
            self._save_attention_weights(alpha, edge_index)

        out = v_j * alpha.view(-1, self.heads, 1)
        return out.view(-1, self.out_channels)

    def _save_attention_weights(self, alpha, edge_index):
        # Custom method to save attention weights
        self._attn_scores = {}
        self._attn_scores = dict(zip(list(edge_index.T), alpha.tolist()))


class HAN(nn.Module):
    def __init__(self, graph, in_channels, out_channels, hidden_channels=128, heads=8):
        super().__init__()
        self.han_conv = HANConv(in_channels, hidden_channels, heads=heads,
                                dropout=0.6, metadata=graph.metadata())
        self.lin = nn.Linear(hidden_channels, out_channels)

    def forward(self, x_dict, edge_index_dict):
        out = self.han_conv(x_dict, edge_index_dict)
        out = self.lin(out['user'])
        return out


class HGT(torch.nn.Module):
    def __init__(self, graph, hidden_channels, out_channels, num_heads, num_layers):
        super().__init__()

        self.lin_dict = torch.nn.ModuleDict()
        for node_type in graph.node_types:
            self.lin_dict[node_type] = Linear(-1, hidden_channels)

        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = HGTConv(hidden_channels, hidden_channels, graph.metadata(),
                           num_heads, group='sum')
            self.convs.append(conv)

        self.lin = Linear(hidden_channels, out_channels)

    def forward(self, x_dict, edge_index_dict):
        x_dict = {
            node_type: self.lin_dict[node_type](x).relu_()
            for node_type, x in x_dict.items()
        }

        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)

        return self.lin(x_dict['user'])


class mod_RGCN(torch.nn.Module):
    def __init__(self, num_relations, node_feature_dims, num_classes, common_dim=128, hidden_dim=128, dropout=0.6):
        super(mod_RGCN, self).__init__()

        # Initial transformation layers for each node type
        self.init_user = nn.Linear(node_feature_dims['user'], common_dim)
        self.init_food = nn.Linear(node_feature_dims['food'], common_dim)
        self.init_ingredient = nn.Linear(node_feature_dims['ingredient'], common_dim)
        self.init_category = nn.Linear(node_feature_dims['category'], common_dim)
        self.init_habit = nn.Linear(node_feature_dims['habit'], common_dim)
        self.lin1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, num_classes)

        # RGCN layers
        self.conv1 = RGCNConv(common_dim, hidden_dim, num_relations)
        self.conv2 = RGCNConv(hidden_dim, hidden_dim, num_relations)
        self.dropout = dropout

    def forward(self, x_dict, edge_index, edge_type):
        x_dict = {key: x.to(torch.float32) for key, x in x_dict.items()}
        # Initial transformations
        x_user = self.init_user(x_dict['user'])
        x_food = self.init_food(x_dict['food'])
        x_ingredient = self.init_ingredient(x_dict['ingredient'])
        x_category = self.init_category(x_dict['category'])
        x_habit = self.init_habit(x_dict['habit'])

        # Concatenate transformed features: This sequence is paramount, or the edges may be misaligned.
        x_all = torch.cat([x_user, x_food, x_ingredient, x_category, x_habit], dim=0)

        # RGCN layers
        x = self.conv1(x_all, edge_index, edge_type)
        x = F.relu(x)
        x = F.dropout(x, training=self.training, p=self.dropout)
        x = self.conv2(x, edge_index, edge_type)

        return x

    def edge_pair_forward(self, edge_index, x):
        src, dst = edge_index
        src = x[src]
        dst = x[dst]
        edge_embedding = torch.cat([src, dst], dim=1)

        edge_embedding = self.lin1(edge_embedding)
        edge_embedding = F.relu(edge_embedding)
        edge_embedding = F.dropout(edge_embedding, training=self.training, p=self.dropout)
        edge_embedding = self.lin2(edge_embedding)

        return F.log_softmax(edge_embedding, dim=1)


class HeteroGraphClassificationModel(torch.nn.Module):
    def __init__(self, hidden_channels, metadata, num_classes=2):
        super().__init__()

        self.conv1 = HeteroConv({
            edge_type: SAGEConv(-1, hidden_channels, add_self_loops=False)
            for edge_type in metadata[1]
        })
        self.conv2 = HeteroConv({
            edge_type: SAGEConv(hidden_channels, hidden_channels, add_self_loops=False)
            for edge_type in metadata[1]
        })

        self.lin1 = Linear(hidden_channels * len(metadata[0]), hidden_channels)
        self.lin2 = Linear(hidden_channels, num_classes)

    def forward(self, x_dict, edge_index_dict, batch_index):
        x_dict = self.conv1(x_dict, edge_index_dict)
        x_dict = {key: F.relu(x) for key, x in x_dict.items()}
        x_dict = self.conv2(x_dict, edge_index_dict)

        # Global pooling
        x_pool = torch.cat([global_mean_pool(x, batch_index[key]) for key, x in x_dict.items()], dim=1)
        x_pool = F.relu(self.lin1(x_pool))
        x_pool = self.lin2(x_pool)
        return F.log_softmax(x_pool, dim=1)


class GAT(torch.nn.Module):
    def __init__(self, num_features, hidden_dim, num_classes, num_heads=8, dropout=0.6):
        super(GAT, self).__init__()
        self.conv1 = GATConv(num_features, hidden_dim, heads=num_heads)
        self.conv2 = GATConv(hidden_dim * num_heads, num_classes, heads=1, concat=False)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)


class RGCN(torch.nn.Module):
    def __init__(self, num_relations, node_feature_dims, num_classes, common_dim=128, hidden_dim=128, dropout=0.6):
        super(RGCN, self).__init__()

        # Initial transformation layers for each node type
        self.init_user = nn.Linear(node_feature_dims['user'], common_dim)
        self.init_food = nn.Linear(node_feature_dims['food'], common_dim)
        self.init_ingredient = nn.Linear(node_feature_dims['ingredient'], common_dim)
        self.init_category = nn.Linear(node_feature_dims['category'], common_dim)
        self.init_habit = nn.Linear(node_feature_dims['habit'], common_dim)

        # RGCN layers
        self.conv1 = RGCNConv(common_dim, hidden_dim, num_relations)
        self.conv2 = RGCNConv(hidden_dim, num_classes, num_relations)
        self.dropout = dropout

    def forward(self, x_dict, edge_index, edge_type):
        x_dict = {key: x.to(torch.float32) for key, x in x_dict.items()}
        # Initial transformations
        x_user = self.init_user(x_dict['user'])
        x_food = self.init_food(x_dict['food'])
        x_ingredient = self.init_ingredient(x_dict['ingredient'])
        x_category = self.init_category(x_dict['category'])
        x_habit = self.init_habit(x_dict['habit'])

        # Concatenate transformed features: This sequence is paramount, or the edges may be misaligned.
        x_all = torch.cat([x_user, x_food, x_ingredient, x_category, x_habit], dim=0)

        # RGCN layers
        x = self.conv1(x_all, edge_index, edge_type)
        x = F.relu(x)
        x = F.dropout(x, training=self.training, p=self.dropout)
        x = self.conv2(x, edge_index, edge_type)

        return F.log_softmax(x, dim=1)


class PromptRGCN(torch.nn.Module):
    def __init__(self, num_relations, node_feature_dims, num_classes, hidden_dim=128, dropout=0.6):
        super(PromptRGCN, self).__init__()
        self.conv1 = RGCNConv(node_feature_dims, hidden_dim, num_relations)
        self.conv2 = RGCNConv(hidden_dim, num_classes, num_relations)
        self.dropout = dropout

    def forward(self, x, edge_index, edge_type):
        x = self.conv1(x, edge_index, edge_type)
        x = F.relu(x)
        x = F.dropout(x, training=self.training, p=self.dropout)

        x = self.conv2(x, edge_index, edge_type)
        return F.log_softmax(x, dim=1)


class SAGE(torch.nn.Module):
    def __init__(self, num_features, num_classes, hidden_dim=128, dropout=0.6):
        super(SAGE, self).__init__()
        self.conv1 = SAGEConv(num_features, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, num_classes)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training, p=self.dropout)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)


class GCN(torch.nn.Module):
    def __init__(self, num_features, num_classes, hidden_dim=128, dropout=0.6):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, num_classes)
        self.dropout = dropout

    def forward(self, x, edge_index, return_embedding=False):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        if return_embedding:
            return x
        x = F.dropout(x, training=self.training, p=self.dropout)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)


class FourLayerGCN(torch.nn.Module):
    def __init__(self, num_features, num_classes, hidden_dim=128, dropout=0.6):
        super(FourLayerGCN, self).__init__()

        # Define four GCN layers
        self.conv1 = GCNConv(num_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        self.conv4 = GCNConv(hidden_dim, num_classes)
        self.dropout = dropout

    def forward(self, x, edge_index):
        # First GCN layer
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training, p=self.dropout)
        # Second GCN layer
        x = F.relu(self.conv2(x, edge_index))
        x = F.dropout(x, training=self.training, p=self.dropout)
        # Third GCN layer
        x = F.relu(self.conv3(x, edge_index))
        x = F.dropout(x, training=self.training, p=self.dropout)
        # Fourth GCN layer
        x = self.conv4(x, edge_index)

        return F.log_softmax(x, dim=1)


class MLP(nn.Module):
    def __init__(self, num_features, num_classes, hidden_dim=128, dropout=0.6):
        super(MLP, self).__init__()

        self.fc1 = nn.Linear(num_features, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 2*hidden_dim)
        self.fc3 = nn.Linear(2*hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, num_classes)
        self.dropout = dropout

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training, p=self.dropout)
        x = F.relu(self.fc2(x))
        x = F.dropout(x, training=self.training, p=self.dropout)
        x = F.relu(self.fc3(x))
        x = F.dropout(x, training=self.training, p=self.dropout)
        x = self.fc4(x)
        return F.sigmoid(x)


class simple_MLP(nn.Module):
    def __init__(self, num_features, num_classes, hidden_dim=128, dropout=0.6):
        super(simple_MLP, self).__init__()

        self.fc1 = nn.Linear(num_features, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        self.dropout = dropout

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training, p=self.dropout)
        x = self.fc2(x)
        return F.sigmoid(x)
