# -*- coding: utf-8 -*-
"""BPR with per-interaction MEG-RW weights."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from recbole.model.general_recommender.bpr import BPR

from meg_rw.reweight import DEFAULT_MEG_RW_FIELD


class WeightedBPR(BPR):
    """RecBole BPR with weighted pairwise loss.

    Expects ``meg_rw_value_field`` (default: ``meg_rw_weight``) to be present in
    train interactions via ``inject_meg_rw_into_train_dataset``.
    """

    def __init__(self, config, dataset):
        super().__init__(config, dataset)
        self.meg_rw_value_field = (
            config.final_config_dict.get("meg_rw_value_field") or DEFAULT_MEG_RW_FIELD
        )

    def calculate_loss(self, interaction):
        user = interaction[self.USER_ID]
        pos_item = interaction[self.ITEM_ID]
        neg_item = interaction[self.NEG_ITEM_ID]

        user_e, pos_e = self.forward(user, pos_item)
        neg_e = self.get_item_embedding(neg_item)
        pos_score = torch.mul(user_e, pos_e).sum(dim=1)
        neg_score = torch.mul(user_e, neg_e).sum(dim=1)

        # Standard BPR: -log(sigmoid(pos-neg)); apply sample weights if available.
        sample_loss = -F.logsigmoid(pos_score - neg_score)
        weights = interaction.interaction.get(self.meg_rw_value_field)
        if weights is None:
            return sample_loss.mean()
        w = weights.float().to(sample_loss.device)
        return (sample_loss * w).sum() / (w.sum() + 1e-12)
