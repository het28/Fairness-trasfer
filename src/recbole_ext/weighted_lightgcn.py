# -*- coding: utf-8 -*-
"""LightGCN with train-edge weights (MEG-RW) via ``inter_matrix(..., value_field=...)``."""

from __future__ import annotations

import numpy as np
import torch

from recbole.model.abstract_recommender import GeneralRecommender
from recbole.model.general_recommender.lightgcn import LightGCN
from recbole.model.init import xavier_uniform_initialization
from recbole.model.loss import BPRLoss, EmbLoss
from recbole.utils import InputType

from recbole_ext.adj_utils import symmetric_norm_adjacency_weighted


class WeightedLightGCN(LightGCN):
    """
    Like RecBole LightGCN, but the symmetric normalized adjacency uses COO ``data`` as weights.

    Provide ``meg_rw_weight`` (or config ``meg_rw_value_field``) on the **train** ``inter_feat``.
    If ``meg_rw_value_field`` is unset, falls back to an unweighted graph (all ones).
    """

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

        self.latent_dim = config["embedding_size"]
        self.n_layers = config["n_layers"]
        self.reg_weight = config["reg_weight"]
        self.require_pow = config["require_pow"]

        self.user_embedding = torch.nn.Embedding(self.n_users, self.latent_dim)
        self.item_embedding = torch.nn.Embedding(self.n_items, self.latent_dim)
        self.mf_loss = BPRLoss()
        self.reg_loss = EmbLoss()

        self.restore_user_e = None
        self.restore_item_e = None

        self.norm_adj_matrix = self.get_norm_adj_mat().to(self.device)

        self.apply(xavier_uniform_initialization)
        self.other_parameter_name = ["restore_user_e", "restore_item_e"]

    def get_norm_adj_mat(self):
        return symmetric_norm_adjacency_weighted(
            self.interaction_matrix, self.n_users, self.n_items
        )
