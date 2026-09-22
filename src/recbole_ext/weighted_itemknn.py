# -*- coding: utf-8 -*-
"""ItemKNN using weighted interaction matrix values."""

from __future__ import annotations

import numpy as np
import torch
import inspect

from recbole.model.general_recommender.itemknn import ItemKNN, ComputeSimilarity

from meg_rw.reweight import DEFAULT_MEG_RW_FIELD


class WeightedItemKNN(ItemKNN):
    """ItemKNN variant that reads interaction matrix with MEG-RW value_field."""

    def __init__(self, config, dataset):
        super(ItemKNN, self).__init__(config, dataset)

        self.k = config["k"]
        # RecBole versions/configs differ on this field. ComputeSimilarity expects
        # exactly {"user", "item"}.
        raw_method = config.final_config_dict.get("knn_method", config.final_config_dict.get("method", "item"))
        self.method = str(raw_method).lower()
        if self.method not in {"user", "item"}:
            self.method = "item"
        self.shrink = config["shrink"] if "shrink" in config else 0.0

        vf = config.final_config_dict.get("meg_rw_value_field") or DEFAULT_MEG_RW_FIELD
        try:
            self.interaction_matrix = dataset.inter_matrix(form="csr", value_field=vf).astype(np.float32)
        except Exception:
            self.interaction_matrix = dataset.inter_matrix(form="csr").astype(np.float32)

        shape = self.interaction_matrix.shape
        assert self.n_users == shape[0] and self.n_items == shape[1]
        _, self.w = ComputeSimilarity(
            self.interaction_matrix,
            topk=self.k,
            shrink=self.shrink,
            **(
                {"method": self.method}
                if "method" in inspect.signature(ComputeSimilarity.__init__).parameters
                else {}
            ),
        ).compute_similarity()

        if self.method == "user":
            self.pred_mat = self.w.dot(self.interaction_matrix).tolil()
        else:
            self.pred_mat = self.interaction_matrix.dot(self.w).tolil()

        self.fake_loss = torch.nn.Parameter(torch.zeros(1))
        self.other_parameter_name = ["w", "pred_mat"]
