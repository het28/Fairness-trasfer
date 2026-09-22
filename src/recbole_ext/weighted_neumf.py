# -*- coding: utf-8 -*-
"""NeuMF with per-interaction MEG-RW weights."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from recbole.model.general_recommender.neumf import NeuMF

from meg_rw.reweight import DEFAULT_MEG_RW_FIELD


class WeightedNeuMF(NeuMF):
    """RecBole NeuMF with weighted pointwise BCE loss."""

    def __init__(self, config, dataset):
        super().__init__(config, dataset)
        self.meg_rw_value_field = (
            config.final_config_dict.get("meg_rw_value_field") or DEFAULT_MEG_RW_FIELD
        )

    def calculate_loss(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]
        label = interaction[self.LABEL].float()

        output = self.forward(user, item)
        # NeuMF forward returns logits; use BCE-with-logits like RecBole NeuMF.
        sample_loss = F.binary_cross_entropy_with_logits(output, label, reduction="none")

        weights = interaction.interaction.get(self.meg_rw_value_field)
        if weights is None:
            return sample_loss.mean()
        w = weights.float().to(sample_loss.device)
        return (sample_loss * w).sum() / (w.sum() + 1e-12)
