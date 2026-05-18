from collections import defaultdict

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from transformers.models.bert.modeling_bert import (
    BertAttention,
    BertIntermediate,
    BertOutput,
)

from fastere.models.components import MultiNonLinearClassifier  # , SelfAttention

# from fastere.models.functions import getPretrainedLMHead
from fastere.utils.focal_loss import focal_loss


class RelAttnLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.attention = BertAttention(config=self.config.model)
        self.crossattention = BertAttention(config=self.config.model)
        self.intermediate = BertIntermediate(config=self.config.model)
        self.output = BertOutput(config=self.config.model)

    def forward(
        self,
        hidden_states,
        attention_mask,
        encoder_hidden_states,
        encoder_attention_mask=None,
    ):
        if encoder_attention_mask is not None:
            encoder_attention_mask = encoder_attention_mask.unsqueeze(1).unsqueeze(1)
        hidden_states = self.crossattention(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
        )[0]
        intermediate_output = self.intermediate(hidden_states)
        return self.output(intermediate_output, hidden_states)


class RelDecoder(nn.Module):
    def __init__(self, theta):
        super().__init__()
        self.config = theta.config
        self.rel_ids = theta.rel_ids
        self.ent_ids = theta.ent_ids
        self.tag_size = len(self.rel_ids)
        self.attn = nn.ModuleList(
            [RelAttnLayer(self.config) for _ in range(self.config.ent_attn_layer_num)]
        )

    def forward(
        self,
        hidden_states,
        attention_mask,
        encoder_hidden_states,
        encoder_attention_mask=None,
    ):
        if isinstance(encoder_hidden_states, list):
            encoder_hidden_states = encoder_hidden_states[1:]
        else:
            encoder_hidden_states = [encoder_hidden_states] * len(self.attn)
        assert len(self.attn) == len(encoder_hidden_states)

        out = [hidden_states]
        for layer, enc_hs in zip(self.attn, encoder_hidden_states):
            hidden_states = layer(
                hidden_states, attention_mask, enc_hs, encoder_attention_mask
            )
            out.append(hidden_states)
        return out


class REModel(pl.LightningModule):
    def __init__(self, theta):
        super().__init__()
        self.log = theta.log
        self.config = theta.config
        self.rel_ids = theta.rel_ids
        self.ent_ids = theta.ent_ids

        hidden_size = self.config.model.hidden_size
        rel_hidden_dim_in = hidden_size * 3
        if self.config.use_rel_mention or self.config.use_filter_in_rel:
            rel_hidden_dim_in = hidden_size * 4

        self.classifier = MultiNonLinearClassifier(
            hidden_size=rel_hidden_dim_in,
            tag_size=len(self.rel_ids),
            layers_num=self.config.rel_mlp_layer_num,
        )

        if self.config.use_rel_attn:
            self.decoder = RelDecoder(theta)

        if self.config.use_length_embedding:
            self.length_embedding = theta.length_embedding

        self.grt_count = 0
        self.hit_count = 0
        self.rel_count = 0
        self.cur_epoch = 0
        self.pre_mode = None
        self.filter_score = {}
        self.rel_type_num = len(self.config.dataset.rels)
        self.ent_type_num = len(self.config.dataset.ents)
        self.entids2tag = {v: k for k, v in enumerate(self.ent_ids)}

        self.remain = 0
        self.total = 0
        self.remain_val = 0
        self.total_val = 0
        self.remain_gold = 0
        self.total_gold = 0

        self.statistic = defaultdict(float)
        self.loss_weight = torch.FloatTensor(
            [self.config.get("na_rel_weight", 1)] + [1] * self.rel_type_num
        )

    def get_triples_label(self, triples, device, ent_groups, mode, epoch):
        triple_labels = torch.zeros(len(ent_groups), device=device, dtype=torch.long)

        if triples is not None and len(triples) == 0:
            return triple_labels

        for i, pair in enumerate(ent_groups):
            b = pair[0]
            for t in triples[b]:
                if t[-1] == -1:
                    break
                if t[:4].tolist() == pair[1:5]:
                    triple_labels[i] = t[4] + 1
                    break

        gt_count = 0
        for triple in triples:
            for t in triple:
                if t[-1] == -1:
                    break
                gt_count += 1

        hit_count = (triple_labels != 0).sum().item()
        rel_count = len(triple_labels)

        if mode != "train" and mode == self.pre_mode:
            self.grt_count += gt_count
            self.hit_count += hit_count
            self.rel_count += rel_count

        return triple_labels

    def prepare(
        self,
        theta,
        hidden_state,
        batch,
        gold_entities=None,
        pred_entities=None,
        mode="train",
        return_loss=False,
    ):
        entities = gold_entities if (mode == "train" or return_loss) else pred_entities

        # ent_ids = theta.ent_ids
        device = hidden_state.device
        input_ids, _, pos, triples, _, _, _ = batch
        bsz, seq_len = input_ids.shape

        val_thres = self.config.use_thres_threshold

        if mode != "train":
            cur_threshold = val_thres
        elif mode == "train" and self.config.use_train_threshold:
            metrics = theta.filter.train_metrics.get("f1", 1e-3)
            cur_threshold = 10000 ** (np.log(metrics**3)) * 0.01
        else:
            cur_threshold = -1

        logits, filter_loss, map_dict = theta.filter(
            hidden_state, entities, triples, mode, current_epoch=theta.current_epoch
        )

        if not self.config.dry_test:
            labels = theta.filter.get_filter_label(entities, triples, logits, map_dict)

            if mode == "train":
                for i in range(self.rel_type_num + 1):
                    self.statistic[f"train_gold_label_{i}_count"] += (
                        (labels == i).sum().item()
                    )
                self.statistic["train_gold_label_count"] += len(labels)
            elif return_loss:
                for i in range(self.rel_type_num + 1):
                    self.statistic[f"val_gold_label_{i}_count"] += (
                        (labels == i).sum().item()
                    )
                self.statistic["val_gold_label_count"] += len(labels)
            else:
                for i in range(self.rel_type_num + 1):
                    self.statistic[f"val_pred_label_{i}_count"] += (
                        (labels == i).sum().item()
                    )
                self.statistic["val_pred_label_count"] += len(labels)

        max_len = self.config.max_seq_len or 300

        ent_groups = []
        bio_tags = []
        rel_input_ids = []
        rel_positional_ids = []
        rel_attention_mask = []
        ent_hidden_states = []
        span_mentions = []

        cls_token = theta.tokenizer.cls_token_id
        sep_token = theta.tokenizer.sep_token_id
        pad_token = theta.tokenizer.pad_token_id
        mask_token = theta.tokenizer.mask_token_id

        for b, entity in enumerate(entities):
            if pos is not None:
                sent_s, sent_e = pos[b, 0], pos[b, 1]
                sent_len = sent_e - sent_s
                if sent_len > max_len:
                    max_len = sent_len + 15
            else:
                sent_len = (input_ids[b] != pad_token).sum().item()
                sent_s, sent_e = 0, sent_len

            ids = [cls_token] + input_ids[b, sent_s:sent_e].tolist() + [sep_token]
            pos_ids = list(range(sent_len + 2))
            masks = [1] * (sent_len + 2)

            bio_ids = np.array([theta.ent_ids[0]] * (sent_len + 2))
            for e in entity:
                bio_ids[e[0] - sent_s + 1] = theta.ent_ids[e[2] + 1]
                bio_ids[e[0] - sent_s + 2 : e[1] - sent_s + 1] = theta.ent_ids[
                    e[2] + self.ent_type_num + 1
                ]
            bio_ids = bio_ids.tolist()

            if mode != "predict" and not self.config.dry_test:
                pred_draft_ent_groups = theta.filter.get_draft_ent_groups(
                    entities, b, map_dict, logits, mode
                )
                gold_draft_ent_groups = theta.filter.get_draft_ent_groups(
                    entities, b, map_dict, labels, mode, pred=logits
                )

                count = len(gold_draft_ent_groups)
                pred_count = len(
                    [
                        1
                        for e in pred_draft_ent_groups
                        if e[2] > self.config.get("alb_test1", 0.5)
                    ]
                )

                if mode == "train" and self.config.use_thres_train:
                    r = theta.filter.train_metrics.get("recall", 0)
                    # f1 = theta.filter.train_metrics.get("f1", 0)
                    ent_count = len(entity)
                    # ent_count_half = int(np.round(ent_count / 2))
                    pair_count = len(gold_draft_ent_groups)
                    pair_count_gold = len(
                        [1 for e in gold_draft_ent_groups if e[2] != 0]
                    )
                    # pred_count_pos = len(
                    #    [1 for e in pred_draft_ent_groups if e[2] > val_thres]
                    # )
                    pred_count_dynamic = int(
                        pair_count - pair_count * r + pred_count * r
                    )
                    strategies = {
                        "default": max(ent_count, pred_count_dynamic),
                        "x2min": max(pair_count_gold * 2 + 1, pred_count_dynamic),
                    }
                    count = int(
                        strategies.get(self.config.use_filter_strategy or "default")
                    )

                draft_ent_groups = (
                    gold_draft_ent_groups[:count]
                    if mode == "train"
                    else pred_draft_ent_groups
                )
            else:
                draft_ent_groups = theta.filter.get_draft_ent_groups(
                    entities, b, map_dict, logits, mode
                )
                pred_count = 0

            marker_mask = 1
            for ent_pair in draft_ent_groups:
                sub_s, sub_e, sub_t = ent_pair[0][:3]
                obj_s, obj_e, obj_t = ent_pair[1][:3]
                score = ent_pair[2]
                filter_score = ent_pair[3] if mode == "train" else ent_pair[2]

                if mode == "test" and len(ids) + 5 > max_len:
                    max_len = len(ids) + 5

                if len(ids) + 5 <= max_len and filter_score > cur_threshold:
                    marker_mask += 1
                    ss_tid = theta.tag_ids[sub_t]
                    os_tid = theta.tag_ids[obj_t]
                    ss_pid = sub_s - sent_s
                    os_pid = obj_s - sent_s

                    if self.config.fix_pid:
                        ss_pid = sub_s - sent_s + 1
                        os_pid = obj_s - sent_s + 1

                    ids += [ss_tid, mask_token, os_tid]
                    pos_ids += [ss_pid, os_pid, os_pid]
                    bio_ids += [
                        theta.ent_ids[sub_t + 1],
                        theta.ent_ids[0],
                        theta.ent_ids[obj_t + 1],
                    ]
                    masks += [marker_mask] * 3

                    if self.config.use_rel_opt2 == "mean":
                        _sub = hidden_state[b, sub_s:sub_e].mean(dim=0)
                        _obj = hidden_state[b, obj_s:obj_e].mean(dim=0)
                        if self.config.use_length_embedding:
                            _sub = _sub + self.length_embedding(
                                torch.tensor(sub_e - sub_s, device=hidden_state.device)
                            )
                            _obj = _obj + self.length_embedding(
                                torch.tensor(obj_e - obj_s, device=hidden_state.device)
                            )
                        ent_hidden_states.append(torch.stack([_sub, _obj]))

                        if self.config.use_filter_in_rel == "span_mention":
                            span_s, span_e = min(sub_s, obj_s), max(sub_e, obj_e)
                            span_mentions.append(
                                hidden_state[b, span_s:span_e].mean(dim=0)
                            )
                    else:
                        raise NotImplementedError(
                            f"use_rel_opt2 '{self.config.use_rel_opt2}' not implemented for head/tail"
                        )

                    ent_groups.append(
                        [b, sub_s, sub_e, obj_s, obj_e, sub_t, obj_t, score]
                    )

            rel_input_ids.append(torch.tensor(ids))
            rel_positional_ids.append(torch.tensor(pos_ids))
            rel_attention_mask.append(torch.tensor(masks))
            bio_tags.append(torch.tensor(bio_ids))

            if mode == "train":
                self.remain += marker_mask - 1
                self.total += (
                    len(gold_draft_ent_groups) if not self.config.dry_test else 0
                )
                self.statistic["train_gold_use_count"] += marker_mask - 1
            elif mode in ("dev", "test") and not self.config.dry_test:
                if return_loss:
                    self.remain_gold += marker_mask - 1
                    self.total_gold += len(pred_draft_ent_groups)
                else:
                    self.remain_val += marker_mask - 1
                    self.total_val += (
                        len(pred_draft_ent_groups) if not self.config.dry_test else 0
                    )

        rel_input_ids = nn.utils.rnn.pad_sequence(
            rel_input_ids, batch_first=True, padding_value=pad_token
        )
        rel_positional_ids = nn.utils.rnn.pad_sequence(
            rel_positional_ids, batch_first=True, padding_value=0
        )
        rel_attention_mask_pad = nn.utils.rnn.pad_sequence(
            rel_attention_mask, batch_first=True, padding_value=0
        )
        bio_tags = nn.utils.rnn.pad_sequence(
            bio_tags, batch_first=True, padding_value=theta.ent_ids[0]
        )

        padding_length = rel_input_ids.shape[1]
        rel_attention_mask_matrix = torch.zeros(
            [bsz, padding_length, padding_length], dtype=torch.long
        )

        for b, m in enumerate(rel_attention_mask):
            cur_len = len(m)
            matrix = []
            for from_mask in m.tolist():
                matrix_i = [
                    1 if (to_mask == 1 or from_mask == to_mask) else 0
                    for to_mask in m.tolist()
                ]
                matrix.append(matrix_i)
            rel_attention_mask_matrix[b, :cur_len, :cur_len] = torch.tensor(matrix)

        rel_attention_mask = rel_attention_mask_matrix.clone()

        rel_input_ids = rel_input_ids.to(device)
        rel_positional_ids = rel_positional_ids.to(device)
        rel_attention_mask = rel_attention_mask.to(device)
        bio_tags = bio_tags.to(device)

        assert rel_positional_ids.max() <= 512 and rel_positional_ids.min() >= 0
        assert rel_input_ids.shape == rel_positional_ids.shape

        rel_hidden_states = []
        sent_ner_loss = torch.tensor(0.0).to(hidden_state.device)

        if ent_groups:
            ent_hidden_states = torch.stack(ent_hidden_states)

            rel_inputs_embeds = self.bert_embeddings_with_bio_tag_embedding(
                theta.plm_model.embeddings.word_embeddings,
                bio_tags=bio_tags,
                input_ids=rel_input_ids,
            )

            plm_model = theta.plm_model
            if self.config.use_rel_prompt:
                batch_size = input_ids.shape[0]
                past_key_values = theta.get_prompt(stage="rel", batch_size=batch_size)
                prefix_attention_mask = torch.ones(
                    batch_size,
                    rel_attention_mask.shape[-1],
                    theta.prompt_len,
                    dtype=torch.long,
                ).to(plm_model.device)
                rel_attention_mask = torch.cat(
                    (prefix_attention_mask, rel_attention_mask), dim=2
                )
            else:
                past_key_values = None

            outputs = plm_model(
                input_ids=None,
                inputs_embeds=rel_inputs_embeds,
                attention_mask=rel_attention_mask.unsqueeze(1).bool(),
                position_ids=rel_positional_ids,
                token_type_ids=torch.zeros(
                    rel_input_ids.shape[:2], dtype=torch.long, device=device
                ),
                output_hidden_states=True,
                past_key_values=past_key_values,
            )
            rel_stage_hs = outputs.hidden_states[-1]

            sent_mask = torch.where(rel_attention_mask_pad == 1, 1.0, 0.0).to(device)

            mask_pos = torch.where(rel_input_ids == theta.tokenizer.mask_token_id)
            rel_hidden_states = rel_stage_hs[mask_pos[0], mask_pos[1]]
            sub_tag_hs = rel_stage_hs[mask_pos[0], mask_pos[1] - 1]
            obj_tag_hs = rel_stage_hs[mask_pos[0], mask_pos[1] + 1]

            if self.config.use_rel_mention:
                mentions = []
                for _bi, _index in zip(mask_pos[0], mask_pos[1]):
                    sub_pos_id = rel_positional_ids[_bi, _index - 1]
                    obj_pos_id = rel_positional_ids[_bi, _index + 1]
                    mention_s = min(sub_pos_id, obj_pos_id)
                    mention_e = max(sub_pos_id, obj_pos_id)
                    mentions.append(
                        rel_stage_hs[_bi, mention_s : mention_e + 1].mean(dim=0)
                    )
                mentions = torch.stack(mentions).to(device)

            if self.config.use_rel_ner and not self.config.dry_test:
                if self.config.use_rel_ner == "no_mask":
                    sent_mask = torch.where(rel_attention_mask_pad > 0, 1, 0).to(device)
                bio_labels = torch.tensor(
                    [[self.entids2tag[bbb] for bbb in bb] for bb in bio_tags.tolist()]
                ).to(device)
                _, sent_ner_loss = theta.ner_model(
                    rel_stage_hs, labels=bio_labels, mask=sent_mask, mode=mode
                )

            sub_hs = ent_hidden_states[:, 0, :] + sub_tag_hs
            obj_hs = ent_hidden_states[:, 1, :] + obj_tag_hs

            if self.config.remove_ner_info:
                sub_hs = sub_tag_hs
                obj_hs = obj_tag_hs

            rel_hidden_states = torch.cat([rel_hidden_states, sub_hs, obj_hs], dim=-1)

            if self.config.use_filter_in_rel:
                sub_pair_proj = theta.filter.sub_proj(ent_hidden_states[:, 0, :])
                obj_pair_proj = theta.filter.obj_proj(ent_hidden_states[:, 1, :])
                ent_pair_hs = torch.cat([sub_pair_proj, obj_pair_proj], dim=-1)
                if self.config.use_filter_in_rel == "span_mention":
                    ent_pair_hs = torch.cat(
                        [ent_pair_hs, torch.stack(span_mentions)], dim=-1
                    )
                ent_pair_hs = theta.filter.filter_entity_pair_net(ent_pair_hs)[1]
                rel_hidden_states = torch.cat([rel_hidden_states, ent_pair_hs], dim=-1)

        if mode != "predict" and not self.config.dry_test:
            triple_labels = self.get_triples_label(
                triples, device, ent_groups, mode=mode, epoch=theta.current_epoch
            )
            assert len(ent_groups) == len(rel_hidden_states) == len(triple_labels)
        else:
            triple_labels = None

        return ent_groups, rel_hidden_states, triple_labels, filter_loss, sent_ner_loss

    def forward(
        self,
        theta,
        batch,
        hidden_state,
        gold_entities=None,
        pred_entities=None,
        return_loss=False,
        mode="train",
        with_score=False,
    ):
        output = self.prepare(
            theta=theta,
            batch=batch,
            hidden_state=hidden_state,
            gold_entities=gold_entities,
            pred_entities=pred_entities,
            mode=mode,
            return_loss=return_loss,
        )

        self.pre_mode = mode
        ent_groups, hidden_output, triple_labels, filter_loss = output[:4]
        sent_ner_loss = output[4] if len(output) > 4 else None

        if len(hidden_output) == 0:
            rel_loss = torch.tensor(0.0).to(hidden_state.device)
            return (
                ([],) if not return_loss else ([], rel_loss, filter_loss, sent_ner_loss)
            )

        logits = self.classifier(hidden_output)
        triples_pred = []
        relation_logits = logits.argmax(dim=-1)
        for i in range(len(ent_groups)):
            rel = relation_logits[i].item()
            triples_pred.append(
                ent_groups[i] + [rel, logits[i]]
                if with_score
                else ent_groups[i] + [rel]
            )

        rel_loss = torch.tensor(0.0).to(logits.device)
        if triple_labels is not None and return_loss:
            new_logits = logits.view(-1, len(self.rel_ids))
            new_labels = triple_labels.view(-1)

            if self.config.use_rel_na_warmup:
                if "_" not in str(self.config.use_rel_na_warmup):
                    self.config.use_rel_na_warmup = "sin_" + str(
                        self.config.use_rel_na_warmup
                    )
                func, warmup_epochs = self.config.use_rel_na_warmup.split("_")
                if func == "sin":
                    warmup_rate = np.sin(
                        (
                            min(1, theta.current_epoch / int(warmup_epochs) + 0.001)
                            * np.pi
                            / 2
                        )
                    )
                else:
                    warmup_rate = 1 - np.cos(
                        (
                            min(1, theta.current_epoch / int(warmup_epochs) + 0.001)
                            * np.pi
                            / 2
                        )
                    )
                self.loss_weight[0] = (
                    float(self.config.get("na_rel_weight", 1)) * warmup_rate
                )

            if self.config.use_rel_focal_loss:
                loss_fct = focal_loss(
                    alpha=None, gamma=2, num_classes=len(self.rel_ids)
                )
                rel_loss = loss_fct(new_logits, new_labels)
            elif self.config.use_rel_loss_sum:
                scale_rate = int(self.config.use_rel_loss_sum)
                loss_fct = nn.CrossEntropyLoss(
                    reduction="sum", weight=self.loss_weight.to(logits.device)
                )
                rel_loss = (
                    loss_fct(new_logits, new_labels)
                    / scale_rate
                    / self.config.batch_size
                    * 16
                )
            else:
                loss_fct = nn.CrossEntropyLoss(
                    reduction="mean", weight=self.loss_weight.to(logits.device)
                )
                rel_loss = loss_fct(new_logits, new_labels)

            return (triples_pred, rel_loss, filter_loss, sent_ner_loss)

        if mode == "predict":
            return (triples_pred, ent_groups)

        return (triples_pred,)

    def log_statistic_train(self):
        self.log(
            "statistic/train_gold_use_count", self.statistic["train_gold_use_count"]
        )
        self.log("filter/rate", self.remain / (self.total + 1e-8))
        self.statistic["train_gold_use_count"] = 0.0
        self.remain = 0
        self.total = 0

    def log_statistic_val(self):
        self.log("filter/rate_val", self.remain_val / (self.total_val + 1e-8))
        self.remain_val = 0
        self.total_val = 0
        self.remain_gold = 0
        self.total_gold = 0

    def log_ent_pair_info(self):
        self.filter_score["precision"] = self.hit_count / (self.rel_count + 1e-8)
        self.filter_score["recall"] = self.hit_count / (self.grt_count + 1e-8)
        self.filter_score["f1"] = (
            2 * self.hit_count / (self.grt_count + self.rel_count + 1e-8)
        )
        self.log("pair/f1", self.filter_score["f1"])
        self.log("pair/precision", self.filter_score["precision"])
        self.log("pair/recall", self.filter_score["recall"])
        self.hit_count = 0
        self.grt_count = 0
        self.rel_count = 0

    def log_filter_rate(self):
        self.log("filter/rate", self.remain / (self.total + 1e-8))
        self.remain = 0
        self.total = 0

    def log_filter_rate_val(self):
        self.log("filter/rate_val", self.remain_val / (self.total_val + 1e-8))
        self.log("filter/rate_gold_ent", self.remain_gold / (self.total_gold + 1e-8))
        self.remain_val = 0
        self.total_val = 0
        self.remain_gold = 0
        self.total_gold = 0

    def bert_embeddings_with_bio_tag_embedding(
        self, word_embeddings, bio_tags, input_ids
    ):
        return word_embeddings(input_ids) + word_embeddings(bio_tags)


def is_overlap(entity):
    for i in range(len(entity)):
        for j in range(i + 1, len(entity)):
            if entity[i][1] > entity[j][0] and entity[i][0] < entity[j][1]:
                return True
    return False
