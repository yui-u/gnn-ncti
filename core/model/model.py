import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as gnn
from torch_geometric.utils import sort_edge_index

from core.common.constants import *
from core.preprocess.reader import (EDGE_LABEL, NOTE_NODE_TYPE_LABEL, NODE_FEATS, NODE_LABELS, NCT_NONCHORD_INDEX, NCT_BOTH_INDEX, NCT_IGNORE_INDEX)


def get_activation_fn(activation_fn_str):
    if activation_fn_str.lower() == 'tanh':
        activation_fn = nn.Tanh()
    elif activation_fn_str.lower() == 'relu':
        activation_fn = nn.ReLU()
    elif activation_fn_str.lower() == 'leaky_relu':
        activation_fn = nn.LeakyReLU()
    elif activation_fn_str.lower() == 'mish':
        activation_fn = nn.Mish()
    else:
        raise NotImplementedError
    return activation_fn


class GraphEncoder(nn.Module):
    def __init__(self, config, device, x_dim):
        super(GraphEncoder, self).__init__()
        self.device = device
        self.model_type = config.gnn_model_type
        self.hidden_size = config.gnn_hidden_size
        self.chroma_dim = 12
        self.dropout_p = config.dropout_p
        self.activation_fn_name = config.gnn_activation_fn
        self.activation_fn = get_activation_fn(config.gnn_activation_fn)
        self.num_gat_heads = config.num_gat_heads
        self.num_message_passing = config.num_message_passing
        self.ablation_edge_feat = config.ablation_edge_feat
        in_channels = self.hidden_size
        out_channels = self.hidden_size
        if 'sage' in self.model_type:
            self._gconv = gnn.SAGEConv
            kwargs = {}
        elif 'gat' in self.model_type:
            in_channels = in_channels * self.num_gat_heads
            self._gconv = gnn.GATv2Conv
            kwargs = {
                'concat': True,
                'heads': self.num_gat_heads,
                'add_self_loops': False,  # already added
            }
            if not self.ablation_edge_feat:
                kwargs['edge_dim'] = 4
        else:
            raise NotImplementedError
        self.pre_enc_layer = nn.Sequential(
            nn.Linear(x_dim, in_channels),
            self.activation_fn,
            nn.Dropout(self.dropout_p),
        )

        self.gconv_list = nn.ModuleList()
        self.num_layers = self.num_message_passing
        for _ in range(self.num_layers):
            self.gconv_list.append(
                self._gconv(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    bias=True,
                    **kwargs
                )
            )
        self.dropout = nn.Dropout(self.dropout_p)
        self.layernorm = gnn.LayerNorm(in_channels=in_channels, mode='graph')
        self.initialize_parameters()

    def initialize_parameters(self):
        gain = nn.init.calculate_gain(self.activation_fn_name)
        for k, v in self.state_dict().items():
            if 'layernorm' not in k:
                if 'weight' in k:
                    nn.init.xavier_uniform_(v, gain=gain)
                if 'bias' in k:
                    nn.init.constant_(v, 0.0)

    def forward(self, graph):
        self_note_edge_pos = torch.where(graph.edge_attr[:, 0].long() == EDGE_LABEL['self_note'])[0]
        self_note_edge_index = torch.index_select(
            graph.edge_index,
            index=self_note_edge_pos,
            dim=1
        )
        self_note_edge_attr = torch.index_select(
            graph.edge_attr,
            index=self_note_edge_pos,
            dim=0
        )
        self_note_edge_attr = torch.cat([
            torch.zeros((self_note_edge_attr.size(0), 3), device=self.device),
            self_note_edge_attr[:, 1].unsqueeze(-1)
        ], dim=-1)
        self_note_edge_attr[:, EDGE_LABEL['self_note']] = 1

        v_edge_pos = torch.where(graph.edge_attr[:, 0].long() == EDGE_LABEL['overlap'])[0]
        v_edge_index = torch.index_select(
            graph.edge_index,
            index=v_edge_pos,
            dim=1
        )
        v_edge_attr = torch.index_select(
            graph.edge_attr,
            index=v_edge_pos,
            dim=0
        )
        v_edge_attr = torch.cat([
            torch.zeros((v_edge_attr.size(0), 3), device=self.device),
            v_edge_attr[:, 1].unsqueeze(-1)
        ], dim=-1)
        v_edge_attr[:, EDGE_LABEL['overlap']] = 1

        h_edge_pos = torch.where(graph.edge_attr[:, 0].long() == EDGE_LABEL['neighbor'])[0]
        h_edge_index = torch.index_select(
            graph.edge_index,
            index=h_edge_pos,
            dim=1
        )
        h_edge_attr = torch.index_select(
            graph.edge_attr,
            index=h_edge_pos,
            dim=0
        )
        h_edge_attr = torch.cat([
            torch.zeros((h_edge_attr.size(0), 3), device=self.device),
            h_edge_attr[:, 1].unsqueeze(-1)
        ], dim=-1)
        h_edge_attr[:, EDGE_LABEL['neighbor']] = 1

        edge_index = torch.cat([
            v_edge_index,
            h_edge_index,
            self_note_edge_index,
        ], dim=1)
        edge_attr = torch.cat([
            v_edge_attr,
            h_edge_attr,
            self_note_edge_attr,
        ], dim=0)

        hidden = self.pre_enc_layer(graph.x)
        for i in range(self.num_message_passing):
            if 'gat' in self.model_type and (not self.ablation_edge_feat):
                hidden = self.layernorm(
                    self.dropout(self.activation_fn(
                        self.gconv_list[i](hidden, edge_index=edge_index, edge_attr=edge_attr))))
            else:
                hidden = self.layernorm(
                    self.dropout(self.activation_fn(
                        self.gconv_list[i](hidden, edge_index=edge_index))))
        return hidden


class NCTModel(nn.Module):
    def __init__(self, config, device):
        super().__init__()
        self.device = device
        self.gnn_model_type = config.gnn_model_type
        self.gnn_hidden_size = config.gnn_hidden_size
        self.mlp_hidden_size = config.mlp_hidden_size
        self.mlp_activation_fn = get_activation_fn(config.mlp_activation_fn)
        self.num_gat_heads = config.num_gat_heads
        self.num_message_passing = config.num_message_passing
        self.dropout_p = config.dropout_p
        self.chroma_dim = 12

        self.ablation_midi = config.ablation_midi
        self.ablation_pc = config.ablation_pc
        self.ablation_beat = config.ablation_beat
        x_dim = 0
        if not self.ablation_pc:
            x_dim += 12
        if not self.ablation_midi:
            x_dim += 128
        if not self.ablation_beat:
            x_dim += 3
        self.graph_encoder = GraphEncoder(config, device, x_dim=x_dim)
        self.activation_fn = get_activation_fn(config.mlp_activation_fn)
        classifier_in_channels = self.gnn_hidden_size
        if 'gat' in self.gnn_model_type:
            classifier_in_channels = classifier_in_channels * self.num_gat_heads
        self.classifier = nn.Sequential(
            nn.Linear(classifier_in_channels, self.mlp_hidden_size),
            nn.Dropout(self.dropout_p),
            self.mlp_activation_fn,
            nn.Linear(self.mlp_hidden_size, 2)
        )

    def forward(self, batch, nct_pos_th=None, eps=1e-7):
        batch_size = len(batch['graph'])
        loss_fn = nn.CrossEntropyLoss(reduction='sum')
        num_items = 0
        gold = []
        pred = []
        all_logit = []
        all_target = []
        all_target_node_ids = []
        for bi in range(batch_size):
            graph = batch['graph'][bi]
            # do not use time nodes
            note_nodes = torch.where(graph.x[:, NODE_LABELS['node_type']] == NOTE_NODE_TYPE_LABEL)[0]
            graph = graph.subgraph(note_nodes)
            ei, ea = sort_edge_index(graph.edge_index, graph.edge_attr)
            graph.edge_index = ei
            graph.edge_attr = ea
            target = graph.y.long()
            target_indices = torch.where(target != NCT_IGNORE_INDEX)[0]
            # pitch class
            pc = graph.x[:, NODE_FEATS['pc']].long()
            pc = torch.where(
                torch.logical_and(0 <= pc, pc < self.chroma_dim),
                pc,
                torch.ones_like(pc) * self.chroma_dim
            )
            pc = F.one_hot(pc, num_classes=self.chroma_dim + 1)[:, :-1].float()
            # midi
            midi = graph.x[:, NODE_FEATS['midi']].long()
            midi = torch.where(
                torch.logical_and(0 <= midi, midi < 128),
                midi,
                torch.ones_like(midi) * 128
            )
            midi = F.one_hot(midi, num_classes=128 + 1)[:, :-1].float()
            # beat (strong downbeat, downbeat, upbeat)
            strong_downbeat = (torch.abs(graph.x[:, NODE_FEATS['beat']] - 1.0) < eps)
            downbeat = torch.logical_and(
                torch.abs(graph.x[:, NODE_FEATS['beat']] - torch.round(graph.x[:, NODE_FEATS['beat']], decimals=0)) < eps,
                torch.logical_not(strong_downbeat))
            upbeat = torch.logical_not(torch.logical_or(strong_downbeat, downbeat))
            beat = torch.stack([strong_downbeat.long(), downbeat.long(), upbeat.long()], dim=1)
            assert 0 == torch.logical_and(strong_downbeat, downbeat).long().sum().item()
            assert 0 == torch.logical_and(strong_downbeat, upbeat).long().sum().item()
            assert 0 == torch.logical_and(downbeat, upbeat).long().sum().item()
            # node feature
            x = []
            if not self.ablation_pc:
                x.append(pc)
            if not self.ablation_midi:
                x.append(midi)
            if not self.ablation_beat:
                x.append(beat)
            graph.x = torch.cat(x, dim=-1)
            hidden = self.graph_encoder(graph)
            hidden = torch.index_select(hidden, index=target_indices, dim=0)
            logit = self.classifier(hidden)
            target = torch.index_select(target, index=target_indices, dim=0).squeeze(-1)
            target = torch.where(
                target == NCT_BOTH_INDEX,
                torch.ones_like(target) * NCT_NONCHORD_INDEX,
                target
            )  # merge BOTH to NONCHORD
            target_node_ids = torch.index_select(note_nodes, index=target_indices, dim=0)
            all_target_node_ids.append(target_node_ids.clone().detach().tolist())
            all_logit.append(logit)
            all_target.append(target)
            gold.append(target.clone().detach().tolist())
            if nct_pos_th is None:
                pred.append(logit.argmax(dim=-1).clone().detach().tolist())
            else:
                prob = F.softmax(logit, dim=-1)
                pred.append(
                    torch.where(
                        nct_pos_th <= prob[:, 1],
                        torch.ones_like(prob[:, 1]),
                        torch.zeros_like(prob[:, 1])
                    ).long().tolist()
                )
            assert len(pred) == len(gold), (len(pred), len(gold))
            num_items += target_indices.size(0)
        all_logit = torch.cat(all_logit, dim=0)
        all_target = torch.cat(all_target, dim=0)
        loss = loss_fn(input=all_logit, target=all_target)
        return {
            LOCAL_LOSS: loss,
            BATCH_SIZE: batch_size,
            'num_items': num_items,
            'pred': pred,
            'gold': gold,
            'node_ids': all_target_node_ids
        }
