import pytorch_lightning as pl
import torch
import torch.nn as nn
from transformers.models.bert.modeling_bert import (
    BertAttention,
    BertIntermediate,
    BertOutput,
)

from fastere.models.components import MultiNonLinearClassifier
from fastere.models.functions import getPretrainedLMHead
from fastere.utils.focal_loss import focal_loss


class EntAttentionLayer(nn.Module):
    def __init__(self, config, word_embeddings_fc, ent_ids):
        super().__init__()
        self.config = config
        self.word_embeddings_fc = word_embeddings_fc
        self.attention = BertAttention(config=self.config.model)
        self.crossattention = BertAttention(config=self.config.model)
        self.intermediate = BertIntermediate(config=self.config.model)
        self.output = BertOutput(config=self.config.model)
        self.ent_range = self.config.ent_attn_range
        self.ent_ids = ent_ids

    def forward(self, hidden_states):
        tag_embeddings = self.word_embeddings_fc()(
            torch.tensor(self.ent_ids, device=hidden_states.device)
        )
        tag_embeddings = tag_embeddings.unsqueeze(0).repeat(
            hidden_states.shape[0], 1, 1
        )

        if self.ent_range > 0:
            attn_mask = torch.ones(
                hidden_states.shape[1],
                hidden_states.shape[1],
                device=hidden_states.device,
            )
            attn_mask = torch.tril(attn_mask, diagonal=self.ent_range) * torch.triu(
                attn_mask, diagonal=-self.ent_range
            )
        else:
            attn_mask = None

        attn_out = self.attention(hidden_states, attention_mask=attn_mask)[0]
        attn_out = self.crossattention(
            hidden_states=attn_out, encoder_hidden_states=tag_embeddings
        )[0]
        intermediate_output = self.intermediate(attn_out)
        return self.output(intermediate_output, attn_out)


class EntDecoder(nn.Module):
    def __init__(self, config, theta):
        super().__init__()
        self.config = config
        self.ent_ids = theta.ent_ids
        hidden_size = config.model.hidden_size
        tag_size = len(self.ent_ids)
        self.ffn = MultiNonLinearClassifier(
            hidden_size, tag_size, layers_num=self.config.ent_mlp_layer_num
        )
        self.ffn_bio = MultiNonLinearClassifier(hidden_size, 3)
        self.attn = nn.ModuleList(
            [
                EntAttentionLayer(
                    config, theta.plm_model.get_input_embeddings, self.ent_ids
                )
                for _ in range(config.ent_attn_layer_num)
            ]
        )

    def forward(self, hidden_states):
        if self.config.dry_test:
            for layer in self.attn:
                hidden_states = layer(hidden_states)
            logits = self.ffn(hidden_states)
            return {
                "logits": logits,
                "logits_out": [None] * len(self.attn) + [logits],
                "attention_out": [None] * len(self.attn) + [hidden_states],
            }

        output = {
            "logits": self.ffn(hidden_states),
            "logits_out": [self.ffn(hidden_states)],
            "attention_out": [hidden_states],
        }
        for layer in self.attn:
            hidden_states = layer(hidden_states)
            logits = self.ffn(hidden_states)
            output["logits"] = logits
            output["logits_out"].append(logits)
            output["attention_out"].append(hidden_states)
        return output


class NERModel(pl.LightningModule):
    def __init__(self, theta):
        super().__init__()
        self.decoder = EntDecoder(theta.config, theta)
        self.config = theta.config
        self.ent_ids = theta.ent_ids
        self.ent_tags_count = len(self.ent_ids)
        self.num_ent_type = len(self.config.dataset.ents)

    def forward(self, hidden_states, labels=None, graph=None, mask=None, mode="train"):
        if mask is None:
            mask = torch.ones(hidden_states.shape[:2], device=hidden_states.device)

        out = self.decoder(hidden_states)
        logits = out["logits"]

        if mode == "test" and self.config.dry_test:
            return logits, None

        if labels is None:
            return logits, torch.tensor(0.0, device=logits.device)

        loss_fct = nn.CrossEntropyLoss(reduction="mean")
        new_logits = logits.view(-1, logits.shape[-1])[mask.reshape(-1) > 0]
        new_labels = labels.reshape(-1).long()[mask.reshape(-1) > 0]
        loss = loss_fct(new_logits, new_labels)

        if self.config.use_ner_focal_loss:
            loss_fct = focal_loss(num_classes=self.ent_tags_count)
            loss = loss_fct(new_logits, new_labels)

        if self.config.use_ner_layer_loss:
            if self.config.use_first_layer_loss:
                layer_logits = out["logits_out"][:-1]
            else:
                layer_logits = out["logits_out"][1:-1]
            for logits_out in layer_logits[::-1]:
                new_logits = logits_out.view(-1, logits_out.shape[-1])[
                    mask.reshape(-1) > 0
                ]
                loss += loss_fct(new_logits, new_labels) * 0.1

        return logits, loss

    def decode_entities(
        self, logits, pos=None, with_score=False, mask=None, mode="train"
    ):
        """Return left-closed, right-open entity spans: [[(start, end, type), ...], ...]"""
        ori_logits = logits.clone()
        is_gt = len(logits.shape) != 3 or logits.shape[-1] != self.ent_tags_count
        if not is_gt:
            logits = torch.argmax(logits, dim=-1)

        entities = []
        bsz, seq_len = logits.shape[0], logits.shape[1]
        for b in range(bsz):
            entity = []
            if pos is not None:
                sent_start, sent_end = pos[b, 0], pos[b, 1]
            else:
                sent_start, sent_end = 1, seq_len - 1
            entity = self.default_decode_strategy(
                logits, with_score, ori_logits, b, entity, sent_start, sent_end
            )
            entities.append(entity)
        return entities

    def default_decode_strategy(
        self, logits, with_score, ori_logits, b, entity, sent_start, sent_end
    ):
        start = False
        for i in range(sent_start, sent_end):
            if 0 < logits[b, i] <= self.num_ent_type:
                start = True
                ent_type_id = logits[b, i].item() - 1
                entity.append(
                    [i, i + 1, ent_type_id, ori_logits[b, i]]
                    if with_score
                    else [i, i + 1, ent_type_id]
                )
            elif start and logits[b, i] > self.num_ent_type:
                entity[-1][1] = i + 1
            else:
                start = False
        return entity

    def hard_decode_strategy(
        self, logits, with_score, ori_logits, b, entity, sent_start, sent_end
    ):
        start = False
        ent_type_id = 0
        for i in range(sent_start, sent_end):
            if 0 < logits[b, i] <= self.num_ent_type:
                start = True
                ent_type_id = logits[b, i].item() - 1
                entity.append(
                    [i, i + 1, ent_type_id, ori_logits[b, i]]
                    if with_score
                    else [i, i + 1, ent_type_id]
                )
            elif start and logits[b, i] == ent_type_id + self.num_ent_type + 1:
                entity[-1][1] = i + 1
            else:
                start = False
        return entity

    def soft_decode_strategy(
        self, logits, with_score, ori_logits, b, entity, sent_start, sent_end
    ):
        start = False
        for i in range(sent_start, sent_end):
            if logits[b, i] > 0:
                ent_type_id = (logits[b, i].item() - 1) % self.num_ent_type
                if start and ent_type_id == entity[-1][2]:
                    entity[-1][1] = i + 1
                else:
                    start = True
                    entity.append(
                        [i, i + 1, ent_type_id, ori_logits[b, i]]
                        if with_score
                        else [i, i + 1, ent_type_id]
                    )
            else:
                start = False
        return entity

    @staticmethod
    def remove_duplicate(entity):
        new_entity = []
        for ent in entity:
            if ent not in new_entity:
                new_entity.append(ent)
        return new_entity
