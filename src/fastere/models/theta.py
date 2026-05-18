import json
import os
from collections import defaultdict

import pytorch_lightning as pl
import torch
import torch.nn as nn

from fastere.data.utils import get_language_map_dict
from fastere.models.components import PrefixEncoder
from fastere.models.entity_pair_filter import FilterModel
from fastere.models.functions import getBertForMaskedLMClass
from fastere.models.ner_model import NERModel
from fastere.models.re_model import REModel
from fastere.utils.metrics import f1_score
from fastere.utils.optimizers import get_optimizer


class Theta(pl.LightningModule):
    def __init__(self, config, data):
        super().__init__()
        self.tag_ids = None
        self.ent_ids = None
        self.rel_ids = None
        self.config = config
        self.tokenizer = data.tokenizer

        self.lr = config.lr
        self.batch_size = config.batch_size
        self.hidden_size = config.model.hidden_size

        ModelClass = getBertForMaskedLMClass(config.model)
        self.plm_model = ModelClass.from_pretrained(
            config.model.model_name_or_path, attn_implementation="eager"
        )

        self.length_embedding = nn.Embedding(512, config.model.hidden_size)
        self.dropout = nn.Dropout(config.model.hidden_dropout_prob)

        if self.config.use_ner_prompt or self.config.use_rel_prompt:
            self.prompt_len = config.prompt_len or 16
            self.n_layer = config.model.num_hidden_layers
            self.n_head = config.model.num_attention_heads
            self.n_embd = config.model.hidden_size // config.model.num_attention_heads
            self.prefix_tokens = torch.arange(self.prompt_len).long()
            projection = config.use_prefix_prompt_projection

            if self.config.use_ner_prompt:
                self.ner_prefix_encoder = PrefixEncoder(
                    num_hidden_layers=self.n_layer,
                    prompt_len=self.prompt_len,
                    prefix_hidden_size=512,
                    hidden_size=self.hidden_size,
                    prefix_projection=projection,
                )
            if self.config.use_rel_prompt:
                self.rel_prefix_encoder = PrefixEncoder(
                    num_hidden_layers=self.n_layer,
                    prompt_len=self.prompt_len,
                    prefix_hidden_size=512,
                    hidden_size=self.hidden_size,
                    prefix_projection=projection,
                )

        self.bce_loss_fn = nn.BCEWithLogitsLoss(reduction="none")

        self.rel_num = len(config.dataset.rels)
        self.ent_num = len(config.dataset.ents)
        self.na_idx = data.rel2id.get(config.dataset.na_label, None)
        self.cur_doc_id = 0
        self.cur_mode = "train"
        self.info_dict = defaultdict(int)

        self.best_f1 = 0
        self.test_f1 = 0
        self.test_f1_plus = 0
        self.test_p = None
        self.test_r = None
        self.ner_f1 = None
        self.rel_f1 = None

        self._val_outputs: list = []
        self._test_outputs: list = []
        self._validate_for_jsonl: bool = False

        self.extend_and_init_additional_tokens()
        self.register_components()

    def get_prompt(self, stage, batch_size):
        prefix_encoder = (
            self.ner_prefix_encoder if stage == "ner" else self.rel_prefix_encoder
        )
        prefix_tokens = (
            self.prefix_tokens.unsqueeze(0)
            .expand(batch_size, -1)
            .to(self.plm_model.device)
        )
        past_key_values = prefix_encoder(prefix_tokens)
        past_key_values = past_key_values.view(
            batch_size, self.prompt_len, self.n_layer * 2, self.n_head, self.n_embd
        )
        past_key_values = self.dropout(past_key_values)
        return past_key_values.permute([2, 0, 3, 1, 4]).split(2)

    def extend_and_init_additional_tokens(self):
        config = self.config
        rels = (
            config.dataset.rels
            if self.na_idx is not None
            else ["NA"] + config.dataset.rels
        )
        ents = config.dataset.ents

        rel_tokens = [f"[R{i}]" for i in range(len(rels))]
        tag_tokens = [f"[S-{e}]" for e in ents] + [f"[E-{e}]" for e in ents]
        ent_tokens = ["[O]"] + [f"[B-{e}]" for e in ents] + [f"[I-{e}]" for e in ents]
        special_tokens = ["[RC]"]

        if config.use_normal_tag:
            tag_tokens += ["[SS]", "[OS]", "[SE]", "[OE]"]

        additional = rel_tokens + ent_tokens + tag_tokens + special_tokens
        cur_size = len(self.tokenizer) + len(additional)
        pad_size = 16 - cur_size % 16
        additional += [f"[PADME{i}]" for i in range(pad_size)]

        self.tokenizer.add_special_tokens({"additional_special_tokens": additional})
        self.plm_model.resize_token_embeddings(len(self.tokenizer))

        self.rel_ids = self.tokenizer.convert_tokens_to_ids(rel_tokens)
        self.ent_ids = self.tokenizer.convert_tokens_to_ids(ent_tokens)
        self.tag_ids = self.tokenizer.convert_tokens_to_ids(tag_tokens)

        with torch.no_grad():
            embeds = self.plm_model.get_input_embeddings().weight
            ace_rel_map, ace_ent_map, tag_map = get_language_map_dict()

            ace_rel_ids = [
                self.tokenizer.encode(
                    ace_rel_map.get(rel, rel), add_special_tokens=False
                )
                for rel in rels
            ]
            for i, rel_id in enumerate(self.rel_ids):
                embeds[rel_id] = embeds[ace_rel_ids[i]].mean(dim=-2)

            ace_ent_ids = [self.tokenizer.encode("outside", add_special_tokens=False)]
            for idx, ent in enumerate(ents):
                ace_ent_ids.insert(
                    idx + 1,
                    self.tokenizer.encode(
                        "begin " + ace_ent_map.get(ent, ent), add_special_tokens=False
                    ),
                )
                ace_ent_ids.append(
                    self.tokenizer.encode(
                        "inside " + ace_ent_map.get(ent, ent), add_special_tokens=False
                    )
                )
            for i, ent_id in enumerate(self.ent_ids):
                embeds[ent_id] = embeds[ace_ent_ids[i]].mean(dim=-2)

            ace_tag_ids = []
            for idx, ent in enumerate(ents):
                ace_tag_ids.insert(
                    idx,
                    self.tokenizer.encode(
                        "start " + ace_ent_map.get(ent, ent), add_special_tokens=False
                    ),
                )
                ace_tag_ids.append(
                    self.tokenizer.encode(
                        "end " + ace_ent_map.get(ent, ent), add_special_tokens=False
                    )
                )
            if config.use_normal_tag:
                for phrase in [
                    "subject start",
                    "object start",
                    "subject end",
                    "object end",
                ]:
                    ace_tag_ids.append(
                        self.tokenizer.encode(phrase, add_special_tokens=False)
                    )
            for i, tag_id in enumerate(self.tag_ids):
                embeds[tag_id] = embeds[ace_tag_ids[i]].mean(dim=-2)

            assert (self.plm_model.get_input_embeddings().weight == embeds).all()

    def register_components(self):
        self.rel_model = REModel(self)
        self.ner_model = NERModel(self)
        self.filter = FilterModel(self)
        self.graph = None

    def forward(self, batch, mode="train"):
        input_ids, attention_mask, pos, triples, ent_maps, sent_mask, span_mask = batch

        if self.config.use_ner_prompt:
            batch_size = input_ids.shape[0]
            past_key_values = self.get_prompt(stage="ner", batch_size=batch_size)
            prefix_attention_mask = torch.ones(
                batch_size, self.prompt_len, dtype=torch.long
            ).to(self.plm_model.device)
            attention_mask = torch.cat((prefix_attention_mask, attention_mask), dim=1)
        else:
            past_key_values = None

        outputs = self.plm_model(
            input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            past_key_values=past_key_values,
        )
        hidden_state = outputs.hidden_states[-1]

        output = {"hidden_state": hidden_state}

        ner_out = self.ner_model(
            hidden_state, labels=ent_maps, graph=self.graph, mask=sent_mask, mode=mode
        )
        ner_logits, ner_loss = ner_out[0], ner_out[1]

        gold_entities = self.ner_model.decode_entities(ent_maps, pos=pos, mode=mode)
        pred_entities = self.ner_model.decode_entities(ner_logits, pos=pos, mode=mode)

        output.update(
            {
                "gold_entities": gold_entities,
                "pred_entities": pred_entities,
                "ner_logits": ner_logits,
                "pos": pos,
            }
        )

        if not (mode == "test" and self.config.dry_test):
            if sum(len(e) for e in gold_entities) > 0:
                rel_output = self.rel_model(
                    theta=self,
                    batch=batch,
                    hidden_state=hidden_state,
                    gold_entities=gold_entities,
                    pred_entities=pred_entities,
                    return_loss=True,
                    mode=mode,
                )
                triples_pred, rel_loss, filter_loss, sent_ner_loss = rel_output
                output["triples_pred_with_gold"] = triples_pred
            else:
                rel_loss = filter_loss = sent_ner_loss = torch.tensor(
                    0.0, device=input_ids.device
                )
                output["triples_pred_with_gold"] = []

        if mode != "train":
            if sum(len(e) for e in pred_entities) > 0:
                rel_output = self.rel_model(
                    theta=self,
                    batch=batch,
                    hidden_state=hidden_state,
                    gold_entities=gold_entities,
                    pred_entities=pred_entities,
                    return_loss=False,
                    mode=mode,
                )
                output["triples_pred"] = rel_output[0]
            else:
                output["triples_pred"] = []

        if mode == "train":
            ner_rate = self.config.ner_rate
            rel_rate = self.config.rel_rate
            filter_rate = self.config.filter_rate
            rel_ner_rate = self.config.rel_ner_rate

            def rate_func(x, max_rate=1):
                if not x or x == 0 or x == "0":
                    return 1
                if self.config.use_convert_warmup_to_steps:
                    return min(
                        max_rate,
                        (self.global_step + 1)
                        / (int(x) / self.config.max_epochs * self.num_training_steps),
                    )
                return min(max_rate, (self.current_epoch + 1) / int(x))

            rel_rate = rate_func(self.config.use_warmup_rel) * rel_rate
            ner_rate = rate_func(self.config.use_warmup_ner) * ner_rate
            filter_rate = rate_func(self.config.use_warmup_filter) * filter_rate

            self.log("loss/ner_loss", ner_loss)
            self.log("loss/rel_loss", rel_loss)
            self.log("loss/filter_loss", filter_loss)
            self.log("loss/sent_ner_loss", sent_ner_loss)

            if self.config.norm_loss_bs:
                norm_gamma = (
                    1 / self.config.global_batch_size * self.config.batch_size * 2
                )
                ner_loss = ner_loss * norm_gamma
                rel_loss = rel_loss * norm_gamma
                filter_loss = (
                    filter_loss
                    if self.config.use_filter_loss_sum
                    else filter_loss * norm_gamma
                )
                sent_ner_loss = sent_ner_loss * norm_gamma

            output["loss"] = (
                ner_loss * ner_rate
                + rel_loss * rel_rate
                + filter_loss * filter_rate
                + sent_ner_loss * rel_ner_rate
            )

        return output

    def training_step(self, batch, batch_idx):
        loss = self(batch, mode="train")["loss"]
        self.log("loss/train_loss", loss)
        for i, pg in enumerate(self.optimizers().param_groups):
            self.log(f"info/lr_{i}", pg["lr"])
        return loss

    def on_train_epoch_end(self):
        self.filter.log_filter_train_metrics()
        self.rel_model.log_filter_rate()
        self.rel_model.log_statistic_train()

    def validation_step(self, batch, batch_idx):
        output = self(batch, mode="dev")
        self._val_outputs.append(self.eval_step_output(batch, output))

    def on_validation_epoch_end(self):
        outputs = self._val_outputs
        f1, p, r = f1_score(outputs, "pred_triples", "gold_triples", slice=3)
        ner_f1, ner_p, ner_r = f1_score(outputs, "pred_entities", "gold_entities")
        rel_f1, rel_p, rel_r = f1_score(
            outputs, "pred_triples", "gold_triples", slice=4
        )
        rel_f1_plus, _, _ = f1_score(outputs, "pred_triples", "gold_triples")
        rel_oracle_f1, rel_oracle_p, rel_oracle_r = f1_score(
            outputs, "pred_triples_with_gold", "gold_triples", slice=4
        )

        self.best_f1 = max(f1, self.best_f1)
        self.log_dict_values({"val_f1": f1, "val/precision": p, "val/recall": r})
        self.log_dict_values({"best_f1": self.best_f1}, on_epoch=True, prog_bar=True)
        self.log_dict_values(
            {"val/ner_f1": ner_f1, "val/ner_p": ner_p, "val/ner_r": ner_r}
        )
        self.log_dict_values(
            {"val/rel_f1": rel_f1, "val/rel_p": rel_p, "val/rel_r": rel_r}
        )
        self.log_dict_values({"val/rel_f1_plus": rel_f1_plus})
        self.log_dict_values(
            {
                "val/rel_oracle_f1": rel_oracle_f1,
                "val/rel_oracle_p": rel_oracle_p,
                "val/rel_oracle_r": rel_oracle_r,
            }
        )
        self.filter.log_filter_val_metrics()
        self.rel_model.log_ent_pair_info()
        self.rel_model.log_filter_rate_val()
        self.rel_model.log_statistic_val()

        if self._validate_for_jsonl:
            data_file = self.config.dataset.get("val", "")
            if data_file:
                out_path = os.path.join(self.config.output_dir, "dev_predictions.jsonl")
                self._write_predictions_jsonl(self._val_outputs, data_file, out_path)
            self._validate_for_jsonl = False

        self._val_outputs.clear()

    def test_step(self, batch, batch_idx):
        output = self(batch, mode="test")
        self._test_outputs.append(self.eval_step_output(batch, output))

    def on_test_epoch_end(self):
        outputs = self._test_outputs
        f1, p, r = f1_score(outputs, "pred_triples", "gold_triples", slice=4)
        f1_plus, p_plus, r_plus = f1_score(outputs, "pred_triples", "gold_triples")
        ner_f1, ner_p, ner_r = f1_score(outputs, "pred_entities", "gold_entities")
        rel_oracle_f1, rel_oracle_p, rel_oracle_r = f1_score(
            outputs, "pred_triples_with_gold", "gold_triples", slice=4
        )

        with open(f"{self.config.output_dir}/output.json", "w") as f:
            json.dump(outputs, f)

        self.test_f1 = f1
        self.test_p = p
        self.test_r = r
        self.test_f1_plus = f1_plus
        self.ner_f1 = ner_f1
        self.rel_f1 = rel_oracle_f1
        self.log_dict_values({"test/rel_f1": f1, "test/rel_p": p, "test/rel_r": r})
        self.log_dict_values(
            {"test/ner_f1": ner_f1, "test/ner_p": ner_p, "test/ner_r": ner_r}
        )
        self.log_dict_values(
            {
                "test/rel_oracle_f1": rel_oracle_f1,
                "test/rel_oracle_p": rel_oracle_p,
                "test/rel_oracle_r": rel_oracle_r,
            }
        )
        self.log_dict_values(
            {
                "test/rel_f1_plus": f1_plus,
                "test/rel_p_plus": p_plus,
                "test/rel_r_plus": r_plus,
            }
        )

        data_file = self.config.dataset.get("test", "")
        if data_file:
            out_path = os.path.join(self.config.output_dir, "test_predictions.jsonl")
            self._write_predictions_jsonl(self._test_outputs, data_file, out_path)

        self._test_outputs.clear()

    def eval_step_output(self, batch, output):
        input_ids, _, pos, triples, ent_maps, sent_mask, _ = batch

        pred_entities = self.get_span_set(input_ids, output["pred_entities"])
        gold_entities = self.get_span_set(input_ids, output["gold_entities"])
        pred_triples, gold_triples = self.get_triple_set(
            input_ids, triples, output, "triples_pred"
        )

        if not self.config.dry_test:
            pred_triples_with_gold = self.get_triple_set(
                input_ids, triples, output, "triples_pred_with_gold", pred_only=True
            )
        else:
            pred_triples_with_gold = []

        # Raw subword positions needed for JSONL output
        pos_list = pos.cpu().tolist()
        pred_entities_raw = [
            [(int(e[0]), int(e[1]), int(e[2])) for e in output["pred_entities"][b]]
            for b in range(input_ids.shape[0])
        ]
        pred_triples_raw = [[] for _ in range(input_ids.shape[0])]
        for t in output.get("triples_pred", []):
            b = int(t[0])
            if int(t[8]) != 0:
                pred_triples_raw[b].append(
                    (int(t[1]), int(t[2]), int(t[3]), int(t[4]), int(t[8]) - 1)
                )

        return {
            "pred_entities": pred_entities,
            "gold_entities": gold_entities,
            "pred_triples": pred_triples,
            "gold_triples": gold_triples,
            "pred_triples_with_gold": pred_triples_with_gold,
            "pos": pos_list,
            "pred_entities_raw": pred_entities_raw,
            "pred_triples_raw": pred_triples_raw,
        }

    # ------------------------------------------------------------------
    # JSONL output helpers
    # ------------------------------------------------------------------

    def _word_subword_maps(self, tokens):
        """Return start2idx and end2idx (word → subword positions, 0-indexed within sentence)."""
        start2idx, end2idx, pos = [], [], 0
        for token in tokens:
            n = len(self.tokenizer.tokenize(token))
            start2idx.append(pos)
            pos += n
            end2idx.append(pos)
        return start2idx, end2idx

    def _subword_to_word_start(
        self, subword_pos, sent_start, start2idx, sentence_start
    ):
        p = subword_pos - sent_start
        for w, s in enumerate(start2idx):
            if s == p:
                return w + sentence_start
        return None

    def _subword_to_word_end(self, subword_pos, sent_start, end2idx, sentence_start):
        p = subword_pos - sent_start
        for w, e in enumerate(end2idx):
            if e == p:
                return w + sentence_start
        return None

    def _write_predictions_jsonl(self, outputs, data_file, out_path):
        """Write predictions in DYGIE JSONL format (one doc per line)."""
        from fastere.data.data_structures import Dataset as DygieDataset

        dataset = DygieDataset(data_file)

        # Collect predictions keyed by (doc_idx, sent_ix)
        pred_by_key = {}
        for batch_out in outputs:
            for b, pos in enumerate(batch_out["pos"]):
                sent_start, sent_end, sent_ix, sent_start_word, doc_idx = pos
                key = (doc_idx, sent_ix)
                if key in pred_by_key:
                    continue

                sent = dataset[doc_idx][sent_ix]
                start2idx, end2idx = self._word_subword_maps(sent.text)

                ner_preds = []
                for s, e, t_idx in batch_out["pred_entities_raw"][b]:
                    w_s = self._subword_to_word_start(
                        s, sent_start, start2idx, sent_start_word
                    )
                    w_e = self._subword_to_word_end(
                        e, sent_start, end2idx, sent_start_word
                    )
                    if w_s is not None and w_e is not None:
                        ner_preds.append([w_s, w_e, self.config.dataset.ents[t_idx]])

                rel_preds = []
                for ss, se, os, oe, r_idx in batch_out["pred_triples_raw"][b]:
                    sw_s = self._subword_to_word_start(
                        ss, sent_start, start2idx, sent_start_word
                    )
                    sw_e = self._subword_to_word_end(
                        se, sent_start, end2idx, sent_start_word
                    )
                    ow_s = self._subword_to_word_start(
                        os, sent_start, start2idx, sent_start_word
                    )
                    ow_e = self._subword_to_word_end(
                        oe, sent_start, end2idx, sent_start_word
                    )
                    if all(x is not None for x in [sw_s, sw_e, ow_s, ow_e]):
                        rel_preds.append(
                            [sw_s, sw_e, ow_s, ow_e, self.config.dataset.rels[r_idx]]
                        )

                pred_by_key[key] = {"ner": ner_preds, "rel": rel_preds}

        with open(out_path, "w") as f:
            for doc_idx, doc in enumerate(dataset.documents):
                js = dataset.js[doc_idx]
                n_sents = len(doc.sentences)
                doc_out = {
                    "doc_key": js["doc_key"],
                    "sentences": js["sentences"],
                    "ner": js.get("ner", [[] for _ in range(n_sents)]),
                    "relations": js.get("relations", [[] for _ in range(n_sents)]),
                    "predicted_ner": [],
                    "predicted_relations": [],
                }
                for sent_ix in range(n_sents):
                    preds = pred_by_key.get((doc_idx, sent_ix), {"ner": [], "rel": []})
                    doc_out["predicted_ner"].append(preds["ner"])
                    doc_out["predicted_relations"].append(preds["rel"])
                f.write(json.dumps(doc_out) + "\n")

    def configure_optimizers(self):
        return get_optimizer(self, self.config)

    def get_triple_set(self, input_ids, triples, output, name, pred_only=False):
        pred_triples = []
        for t in output[name]:
            if t[8] != 0:
                sub_token = self.tokenizer.decode(input_ids[t[0], t[1] : t[2]])
                obj_token = self.tokenizer.decode(input_ids[t[0], t[3] : t[4]])
                rel_type = self.config.dataset.rels[t[8] - 1]
                sub_type = self.config.dataset.ents[t[5]]
                obj_type = self.config.dataset.ents[t[6]]
                pred_triples.append(
                    (t[0], sub_token, obj_token, rel_type, sub_type, obj_type)
                )

        if pred_only:
            return pred_triples

        gold_triples = []
        for b in range(input_ids.shape[0]):
            for t in triples[b]:
                if t[4] != -1:
                    sub_token = self.tokenizer.decode(input_ids[b, t[0] : t[1]])
                    obj_token = self.tokenizer.decode(input_ids[b, t[2] : t[3]])
                    rel_type = self.config.dataset.rels[t[4]]
                    sub_type = self.config.dataset.ents[t[5]]
                    obj_type = self.config.dataset.ents[t[6]]
                    gold_triples.append(
                        (b, sub_token, obj_token, rel_type, sub_type, obj_type)
                    )
                else:
                    break
        return pred_triples, gold_triples

    def get_span_set(self, input_ids, entities):
        entities_token = []
        for b in range(input_ids.shape[0]):
            for e in entities[b]:
                ent_token = self.tokenizer.decode(input_ids[b, e[0] : e[1]])
                ent_type = self.config.dataset.ents[e[2]]
                entities_token.append((b, ent_token, ent_type))
        return entities_token

    def log_dict_values(self, d, **kwargs):
        for k, v in d.items():
            self.log(k, v, **kwargs)

    def get_rel_tag_embeddings(self, with_na=False, with_grad=True, device=None):
        device = device or self.device
        rel_tag_embeddings = self.plm_model.get_input_embeddings().weight[
            torch.tensor(self.rel_ids, device=self.device)
        ]
        if not with_na:
            rel_tag_embeddings = rel_tag_embeddings[1:]
        if not with_grad:
            rel_tag_embeddings = rel_tag_embeddings.detach()
        return rel_tag_embeddings
