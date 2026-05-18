import itertools
import logging
import os
import random

# import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
# from sklearn.metrics import f1_score, precision_score, recall_score

from fastere.models.components import MultiNonLinearClassifier  # , SelfAttention
from fastere.utils.focal_loss import focal_loss
from fastere.utils.metrics import f1_score_simple

logger = logging.getLogger(__name__)


class FilterModel(pl.LightningModule):
    def __init__(self, theta) -> None:
        super().__init__()
        self.config = theta.config
        self.enable = self.config.use_filter and self.config.filter_rate > 0
        self.log = theta.log
        self.pred = None
        self.labels = None
        self.pred_val = None
        self.labels_val = None
        hidden_size = self.config.model.hidden_size
        dropout_rate = self.config.filter_dropout_rate

        self.tag_size = len(self.config.dataset.rels) + 1
        self.get_bio_tag_embedding = lambda d: theta.plm_model.get_input_embeddings()(
            torch.tensor(theta.ent_ids, device=d)
        )

        if self.config.use_length_embedding:
            self.length_embedding = theta.length_embedding

        self.rel_ids = theta.rel_ids
        self.rel_type_num = len(self.config.dataset.rels)
        self.ent_type_num = len(self.config.dataset.ents)
        self.sub_proj = nn.Linear(hidden_size, hidden_size)
        self.obj_proj = nn.Linear(hidden_size, hidden_size)
        self.use_span_mention = self.config.use_span_mention

        filter_input_size = hidden_size * 2
        if self.config.get("use_filter_sent_hs"):
            filter_input_size += hidden_size
            self.sent_proj = nn.Linear(hidden_size, hidden_size)
        if self.use_span_mention:
            filter_input_size += hidden_size

        hidden_dim = hidden_size if self.config.use_filter_in_rel else None

        self.filter_entity_pair_net = MultiNonLinearClassifier(
            filter_input_size,
            layers_num=self.config.get("filter_mlp_layer_num", 1),
            dropout_rate=dropout_rate,
            hidden_dim=hidden_dim,
            tag_size=self.tag_size,
            use_norm=self.use_span_mention,
            return_hidden_size=bool(self.config.use_filter_in_rel),
        )

        if self.config.get("use_filter_proj_norm"):
            self.sub_noise = nn.Parameter(torch.randn(hidden_size))
            self.obj_noise = nn.Parameter(torch.randn(hidden_size))
            nn.init.normal_(self.sub_proj.weight, std=0.01)
            nn.init.normal_(self.obj_proj.weight, std=0.01)
            nn.init.constant_(self.sub_proj.bias, 0)
            nn.init.constant_(self.obj_proj.bias, 0)

        data_dir = self.config.dataset.get("data_dir", "")
        corres_path = os.path.join(data_dir, "ent_rel_corres.data") if data_dir else ""
        if corres_path and os.path.exists(corres_path):
            self.hard_filter_table = torch.load(corres_path, weights_only=True).sum(
                dim=-1
            )
        else:
            self.hard_filter_table = torch.ones(self.ent_type_num, self.ent_type_num)

        self.dropout_hidden_state = nn.Dropout(0.1)
        self.loss_weight = torch.FloatTensor(
            [self.config.get("na_filter_weight", 1)] + [1] * self.rel_type_num
        )

        self.train_metrics = {"f1": 0, "precision": 0, "recall": 0}
        self.val_metrics = {"f1": 0, "precision": 0, "recall": 0}

    def convert_bij_to_index(self, bij, entities):
        batch_sqrt = [len(e) * len(e) for e in entities]
        batch_start = [0] + list(itertools.accumulate(batch_sqrt))[:-1]
        b, i, j = bij
        batch_num = len(entities[b])
        return batch_start[b] + i * batch_num + j

    def get_filter_label(self, entities, triples, logits, map_dict):
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)
        for b, triple in enumerate(triples):
            for t in triple:
                if t[0] == -1:
                    continue
                i = map_dict.get((b, t[0].item(), t[1].item()))
                j = map_dict.get((b, t[2].item(), t[3].item()))
                if i is None or j is None:
                    continue
                index = self.convert_bij_to_index((b, i, j), entities)
                labels[index] = t[4] + 1
                if self.config.get("use_filter_label_enhance"):
                    index = self.convert_bij_to_index((b, j, i), entities)
                    labels[index] = 1
        return labels

    def forward(
        self, hidden_state, entities, triples=None, mode="train", current_epoch=None
    ):
        logits_list = []
        map_dict = {}

        hidden_state = self.dropout_hidden_state(hidden_state)

        for i, ents in enumerate(entities):
            if not ents:
                continue

            for j, e in enumerate(ents):
                map_dict[(i, e[0], e[1])] = j

            if not self.config.use_rel_opt2 or self.config.use_rel_opt2 == "head":
                ent_hs = torch.stack([hidden_state[i, ent[0]] for ent in ents])
            elif self.config.use_rel_opt2 == "mean":
                ent_hs = torch.stack(
                    [hidden_state[i, ent[0] : ent[1]].mean(dim=0) for ent in ents]
                )
            elif self.config.use_rel_opt2 == "max":
                ent_hs = torch.stack(
                    [hidden_state[i, ent[0] : ent[1]].max(dim=0)[0] for ent in ents]
                )
            else:
                raise NotImplementedError(f"use_rel_opt2: {self.config.use_rel_opt2}")

            if self.config.use_length_embedding:
                length_embeds = self.length_embedding(
                    torch.tensor(
                        [ent[1] - ent[0] for ent in ents], device=hidden_state.device
                    )
                )
                ent_hs = ent_hs + length_embeds

            ent_num, _ = ent_hs.shape
            ent_hs_x = self.sub_proj(ent_hs)
            ent_hs_y = self.obj_proj(ent_hs)

            if self.config.get("use_filter_proj_norm"):
                ent_hs_x = ent_hs_x + self.sub_noise
                ent_hs_y = ent_hs_y - self.obj_noise

            if self.use_span_mention:
                ent_pair_features = []
                for sub_i, sub in enumerate(ents):
                    for obj_i, obj in enumerate(ents):
                        span_s, span_e = min(sub[0], obj[0]), max(sub[1], obj[1])
                        span_mention = hidden_state[i, span_s:span_e].mean(dim=0)
                        ent_pair_features.append(
                            torch.cat([ent_hs_x[sub_i], ent_hs_y[obj_i], span_mention])
                        )
                ent_pair_logits = self.filter_entity_pair_net(
                    torch.stack(ent_pair_features)
                )
                if self.config.use_filter_in_rel:
                    ent_pair_logits, _ = ent_pair_logits
            else:
                ent_hs_x = ent_hs_x.unsqueeze(0).repeat(ent_num, 1, 1)
                ent_hs_y = ent_hs_y.unsqueeze(1).repeat(1, ent_num, 1)

                if self.config.get("use_filter_sent_hs"):
                    sent_hs = (
                        hidden_state[i][0].unsqueeze(0).repeat(ent_num, ent_num, 1)
                    )
                    pair_input = torch.cat([sent_hs, ent_hs_x, ent_hs_y], dim=-1)
                else:
                    pair_input = torch.cat([ent_hs_x, ent_hs_y], dim=-1)

                ent_pair_logits = self.filter_entity_pair_net(pair_input)
                if self.config.use_filter_in_rel:
                    ent_pair_logits, _ = ent_pair_logits
                ent_pair_logits = ent_pair_logits.view(-1, self.tag_size).squeeze(-1)

            logits_list.append(ent_pair_logits)

        if not logits_list:
            device = hidden_state.device
            return None, torch.tensor(0.0, device=device), {}

        logits = torch.cat(logits_list, dim=0)
        loss = torch.tensor(0.0, device=logits.device)

        if triples is not None:
            labels = self.get_filter_label(entities, triples, logits, map_dict)

            if self.config.get("use_filter_focal_loss"):
                scale_rate = int(self.config.use_filter_loss_sum) * 100
                loss_fct = focal_loss(
                    alpha=self.config.get("use_focal_alpha"),
                    gamma=2,
                    num_classes=self.tag_size,
                    size_average=False,
                )
                loss = loss_fct(logits, labels.long()) / scale_rate
            elif self.config.use_filter_loss_sum and str(
                self.config.use_filter_loss_sum
            ) not in ("0", ""):
                scale_rate = int(self.config.use_filter_loss_sum)
                loss_fct = nn.CrossEntropyLoss(
                    reduction="sum", weight=self.loss_weight.to(logits.device)
                )
                loss = loss_fct(logits, labels.long()) / scale_rate
            else:
                loss_fct = nn.CrossEntropyLoss(
                    reduction="mean", weight=self.loss_weight.to(logits.device)
                )
                loss = loss_fct(logits, labels.long())

            pred = torch.argmax(logits, dim=-1)
            if mode == "train":
                self.pred = pred if self.pred is None else torch.cat([self.pred, pred])
                self.labels = (
                    labels if self.labels is None else torch.cat([self.labels, labels])
                )
            elif mode in ("dev", "test"):
                self.pred_val = (
                    pred if self.pred_val is None else torch.cat([self.pred_val, pred])
                )
                self.labels_val = (
                    labels
                    if self.labels_val is None
                    else torch.cat([self.labels_val, labels])
                )

        logits = logits.softmax(dim=-1)[:, 1:].sum(dim=-1)
        return logits, loss, map_dict

    def log_filter_train_metrics(self):
        if self.pred is not None and self.labels is not None:
            f1, p, r = f1_score_simple(
                self.labels.cpu().numpy(), self.pred.cpu().numpy()
            )
            self.train_metrics = {
                "f1": float(f1),
                "precision": float(p),
                "recall": float(r),
            }
            self.log("filter/f1", self.train_metrics["f1"])
            self.log("filter/precision", self.train_metrics["precision"])
            self.log("filter/recall", self.train_metrics["recall"])
            self.pred = None
            self.labels = None
            if self.config.debug:
                logger.debug("Filter train F1: %.4f", self.train_metrics["f1"])

    def log_filter_val_metrics(self):
        if self.pred_val is not None and self.labels_val is not None:
            f1, p, r = f1_score_simple(
                self.labels_val.cpu().numpy(), self.pred_val.cpu().numpy()
            )
            self.val_metrics = {
                "f1": float(f1),
                "precision": float(p),
                "recall": float(r),
            }
            self.log("filter/f1_val", self.val_metrics["f1"])
            self.log("filter/precision_val", self.val_metrics["precision"])
            self.log("filter/recall_val", self.val_metrics["recall"])
            self.pred_val = None
            self.labels_val = None
            if self.config.debug:
                logger.debug("Filter val F1: %.4f", self.val_metrics["f1"])

    def get_draft_ent_groups(
        self, entities, batch_idx, map_dict, logits, mode, pred=None
    ):
        ent_groups = []
        pairs = list(itertools.permutations(entities[batch_idx], 2))
        if not pairs:
            return ent_groups

        if self.enable:
            random.shuffle(pairs)

        for sub_pos, obj_pos in pairs:
            i = map_dict.get((batch_idx, sub_pos[0], sub_pos[1]))
            j = map_dict.get((batch_idx, obj_pos[0], obj_pos[1]))
            if i is None or j is None or i == j:
                continue
            index = self.convert_bij_to_index((batch_idx, i, j), entities)
            score = logits[index].item()

            if self.config.get("use_hard_filter_train") or (
                mode != "train" and self.config.dataset.name == "ace2005"
            ):
                sub_type, obj_type = sub_pos[2], obj_pos[2]
                if self.hard_filter_table[sub_type, obj_type] == 0:
                    continue

            if pred is not None:
                ent_groups.append((sub_pos, obj_pos, score, pred[index].item()))
            else:
                ent_groups.append((sub_pos, obj_pos, score))

        if pred is not None:
            use_negative = self.config.use_negative
            if use_negative == "refine":

                def key(a):
                    return bool(a[2]), a[3]
            elif use_negative == "simple":

                def key(a):
                    return bool(a[2]), -a[3]
            elif use_negative == "random":

                def key(a):
                    return bool(a[2])
            else:

                def key(a):
                    return a[2], a[3]

            ent_groups = sorted(ent_groups, key=key, reverse=True)

            if self.config.get("use_drop_top"):
                positive_count = sum(1 for g in ent_groups if g[2] > 0)
                _t = float("0." + self.config.use_drop_top)
                if (
                    len(ent_groups) > positive_count
                    and ent_groups[positive_count][3] > _t
                ):
                    ent_groups.pop(positive_count)
        else:
            ent_groups = sorted(ent_groups, key=lambda a: a[2], reverse=True)

        return ent_groups

    def hard_filter(self):
        pass
