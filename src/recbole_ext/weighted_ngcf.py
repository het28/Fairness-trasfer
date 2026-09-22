# -*- coding: utf-8 -*-
"""NGCF with MEG-RW weighted adjacency."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from recbole.model.abstract_recommender import GeneralRecommender
from recbole.model.general_recommender.ngcf import NGCF
from recbole.model.init import xavier_normal_initialization
from recbole.model.layers import BiGNNLayer, SparseDropout
from recbole.model.loss import BPRLoss, EmbLoss
from recbole.utils import InputType

from recbole_ext.adj_utils import symmetric_norm_adjacency_weighted


class WeightedNGCF(NGCF):
    """Same as RecBole NGCF; ``get_norm_adj_mat`` uses train edge weights."""

    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        GeneralRecommender.__init__(self, config, dataset)

        vf = config.final_config_dict.get("meg_rw_value_field")
        if vf:
            self.interaction_matrix = (
                dataset.inter_matrix(form="coo", value_field=vf).astype(np.float32)
            )
        else:
            self.interaction_matrix = dataset.inter_matrix(form="coo").astype(np.float32)

        self.embedding_size = config["embedding_size"]
        self.hidden_size_list = config["hidden_size_list"]
        self.hidden_size_list = [self.embedding_size] + self.hidden_size_list
        self.node_dropout = config["node_dropout"]
        self.message_dropout = config["message_dropout"]
        self.reg_weight = config["reg_weight"]

        self.sparse_dropout = SparseDropout(self.node_dropout)
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_size)
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_size)
        self.emb_dropout = nn.Dropout(self.message_dropout)
        self.GNNlayers = torch.nn.ModuleList()
        for idx, (input_size, output_size) in enumerate(
            zip(self.hidden_size_list[:-1], self.hidden_size_list[1:])
        ):
            self.GNNlayers.append(BiGNNLayer(input_size, output_size))
        self.mf_loss = BPRLoss()
        self.reg_loss = EmbLoss()

        self.restore_user_e = None
        self.restore_item_e = None

        self.norm_adj_matrix = self.get_norm_adj_mat().to(self.device)
        self.eye_matrix = self.get_eye_mat().to(self.device)

        self.apply(xavier_normal_initialization)
        self.other_parameter_name = ["restore_user_e", "restore_item_e"]

    def get_norm_adj_mat(self):
        return symmetric_norm_adjacency_weighted(
            self.interaction_matrix, self.n_users, self.n_items
        )
