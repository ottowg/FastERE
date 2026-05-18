import numpy as np
import torch


def _assign_ner_to_layers(entities, num_layers, label_preference):
    """Greedily assign entities to BIO layers without conflicts within a layer.

    Sort key: (start_sent, -span_length, pref_rank, label) so earlier-start,
    longer-span, preferred-label entities are placed first.  Two spans conflict
    when they overlap: s1 <= e2 and s2 <= e1 (inclusive end indices).

    Returns a list of (ner, layer_idx) tuples.  Entities that cannot fit in any
    layer are dropped with a printed warning.
    """
    pref_rank = {label: i for i, label in enumerate(label_preference)}

    def sort_key(ner):
        span_len = ner.span.end_sent - ner.span.start_sent
        rank = pref_rank.get(ner.label, len(label_preference))
        return (ner.span.start_sent, -span_len, rank, ner.label)

    sorted_entities = sorted(entities, key=sort_key)

    layer_occupants = [[] for _ in range(num_layers)]
    assignments = []

    for ner in sorted_entities:
        s1, e1 = ner.span.start_sent, ner.span.end_sent
        placed = False
        for layer_idx in range(num_layers):
            conflict = any(
                s1 <= e2 and s2 <= e1
                for occ in layer_occupants[layer_idx]
                for s2, e2 in [(occ.span.start_sent, occ.span.end_sent)]
            )
            if not conflict:
                layer_occupants[layer_idx].append(ner)
                assignments.append((ner, layer_idx))
                placed = True
                break
        if not placed:
            print(
                f"Warning: entity [{s1},{e1}] '{ner.label}' could not fit in any of "
                f"{num_layers} NER layers and was dropped."
            )

    return assignments


def convert_dataset_to_samples(dataset, config, tokenizer, is_test=False):
    """Convert a Dataset to padded tensors ready for DataLoader."""
    samples = []
    num_ner = 0
    max_len = 0
    max_ner = 0

    ner2id = {name: idx for idx, name in enumerate(config.dataset.ents)}
    rel2id = {name: idx for idx, name in enumerate(config.dataset.rels)}

    max_seq_len = config.get("max_seq_len", config.dataset.max_seq_len)
    context_window = config.get("context_window", 0)
    # max_span_length = config.get("max_span_length", 10)

    if context_window and context_window > max_seq_len - 2:
        print(
            f"context_window {context_window} > max_seq_len {max_seq_len}, clamping to {max_seq_len - 2}"
        )
        context_window = max_seq_len - 2

    split = config.dataset.get("split", 0)
    if split == 0:
        data_range = (0, len(dataset))
    elif split == 1:
        data_range = (0, int(len(dataset) * 0.9))
    elif split == 2:
        data_range = (int(len(dataset) * 0.9), len(dataset))
    else:
        data_range = (0, len(dataset))

    cores_table = torch.zeros([len(ner2id), len(ner2id), len(rel2id)])

    for c, doc in enumerate(dataset):
        if c < data_range[0] or c >= data_range[1]:
            continue
        for i, sent in enumerate(doc):
            num_ner += len(sent.ner)
            sample = {"tokens": sent.text}
            sent_length = len(sent.text)

            if context_window and sent_length > context_window:
                print(f"Long sentence: {sample} {sent_length}")

            max_len = max(max_len, sent_length)
            max_ner = max(max_ner, len(sent.ner))

            start2idx = []
            end2idx = []
            tokenized_tokens = []
            for token in sample["tokens"]:
                tok = tokenizer.tokenize(token)
                tokenized_tokens += tok
                start2idx.append(len(tokenized_tokens) - len(tok))
                end2idx.append(len(tokenized_tokens))

            sent_start = 0
            sent_end = len(tokenized_tokens)

            ner_left = []
            ner_right = []
            if context_window:
                add_left = (context_window - len(tokenized_tokens)) // 2
                left_tokens = []
                left_sent_idx = sent.sentence_ix - 1
                while left_sent_idx >= 0 and add_left > 0:
                    left_token_to_add = tokenizer.tokenize(
                        " ".join(doc[left_sent_idx].text)
                    )[-add_left:]
                    left_tokens = left_token_to_add + left_tokens
                    add_left -= len(left_token_to_add)
                    ner_left.extend(doc[left_sent_idx].ner)
                    left_sent_idx -= 1

                sent_start = len(left_tokens)
                tokenized_tokens = left_tokens + tokenized_tokens

                add_right = context_window - len(tokenized_tokens)
                right_tokens = []
                right_sent_idx = sent.sentence_ix + 1
                while right_sent_idx < len(doc) and add_right > 0:
                    right_token_to_add = tokenizer.tokenize(
                        " ".join(doc[right_sent_idx].text)
                    )[:add_right]
                    right_tokens = right_tokens + right_token_to_add
                    add_right -= len(right_token_to_add)
                    ner_right.extend(doc[right_sent_idx].ner)
                    right_sent_idx += 1

                tokenized_tokens = tokenized_tokens + right_tokens

            if len(tokenized_tokens) < max_seq_len - 2:
                attention_mask = torch.zeros(max_seq_len, dtype=torch.long)
                attention_mask[: len(tokenized_tokens) + 2] = 1
                tokenized_tokens = (
                    [tokenizer.cls_token] + tokenized_tokens + [tokenizer.sep_token]
                )
                tokenized_tokens += [tokenizer.pad_token] * (
                    max_seq_len - len(tokenized_tokens)
                )
            else:
                attention_mask = torch.ones(max_seq_len, dtype=torch.long)
                tokenized_tokens = (
                    [tokenizer.cls_token]
                    + tokenized_tokens[: max_seq_len - 2]
                    + [tokenizer.sep_token]
                )

            assert len(tokenized_tokens) == len(attention_mask) == max_seq_len

            input_ids = torch.tensor(
                tokenizer.convert_tokens_to_ids(tokenized_tokens), dtype=torch.long
            )
            assert sum(input_ids != tokenizer.pad_token_id) == sum(attention_mask != 0)

            sent_start += 1
            sent_end += sent_start
            if sent_end > max_seq_len:
                sent_end = max_seq_len - 1

            span_mask = np.zeros((max_seq_len, max_seq_len), dtype=np.int16)

            start2idx = [mi + sent_start for mi in start2idx]
            end2idx = [mi + sent_start for mi in end2idx]

            ent_type = {}
            stacked = config.ner_stacked and config.get("num_ner_layers", 1) > 1
            num_ner_layers = config.get("num_ner_layers", 1) if stacked else 1
            if stacked:
                ent_maps = torch.zeros(
                    len(tokenized_tokens), num_ner_layers, dtype=torch.int16
                )
            else:
                ent_maps = torch.zeros(len(tokenized_tokens), dtype=torch.int16)
            ent_maps_2d = torch.zeros(
                len(tokenized_tokens), len(tokenized_tokens), dtype=torch.int8
            )

            if config.use_cross_ner:
                added_text = []
                for ner in sent.ner + ner_left + ner_right:
                    text = " ".join(ner.span.text)
                    if text in added_text:
                        continue
                    ner_tokens = tokenizer.tokenize(text)
                    ner_s_list = find_subarray_position(tokenized_tokens, ner_tokens)
                    if not ner_s_list:
                        continue
                    for ner_s in ner_s_list:
                        ner_e = ner_s + len(ner_tokens)
                        if ner_s >= max_seq_len or ner_e >= max_seq_len:
                            continue
                        ent_maps[ner_s] = ner2id[ner.label] + 1
                        ent_maps[ner_s + 1 : ner_e] = (
                            ner2id[ner.label] + len(ner2id) + 1
                        )
                        ent_maps_2d[ner_s, ner_e] = ner2id[ner.label] + 1
                        ent_type[(ner_s, ner_e)] = ner2id[ner.label]
                    added_text.append(text)
            else:
                if stacked:
                    label_pref = list(config.get("stacked_ner_label_preference") or [])
                    assignments = _assign_ner_to_layers(
                        sent.ner, num_ner_layers, label_pref
                    )
                    for ner, layer_idx in assignments:
                        ent_s = start2idx[ner.span.start_sent]
                        ent_e = end2idx[ner.span.end_sent]
                        if ent_s >= max_seq_len or ent_e >= max_seq_len:
                            continue
                        ent_maps[ent_s, layer_idx] = ner2id[ner.label] + 1
                        ent_maps[ent_s + 1 : ent_e, layer_idx] = (
                            ner2id[ner.label] + len(ner2id) + 1
                        )
                        ent_maps_2d[ent_s, ent_e] = ner2id[ner.label] + 1
                        assert ent_s < ent_e
                        if (ent_s, ent_e) not in ent_type:
                            ent_type[(ent_s, ent_e)] = ner2id[ner.label]
                else:
                    for ner in sent.ner:
                        ent_s = start2idx[ner.span.start_sent]
                        ent_e = end2idx[ner.span.end_sent]
                        if ent_s >= max_seq_len or ent_e >= max_seq_len:
                            continue
                        ent_maps[ent_s] = ner2id[ner.label] + 1
                        ent_maps[ent_s + 1 : ent_e] = (
                            ner2id[ner.label] + len(ner2id) + 1
                        )
                        ent_maps_2d[ent_s, ent_e] = ner2id[ner.label] + 1
                        assert ent_s < ent_e
                        ent_type[(ent_s, ent_e)] = ner2id[ner.label]

            triples = set()
            for rel in sent.relations:
                sub_s = start2idx[rel.pair[0].start_sent]
                sub_e = end2idx[rel.pair[0].end_sent]
                obj_s = start2idx[rel.pair[1].start_sent]
                obj_e = end2idx[rel.pair[1].end_sent]
                if any(x >= max_seq_len for x in [sub_s, sub_e, obj_s, obj_e]):
                    continue
                sub_type = ent_type.get((sub_s, sub_e), 0) # @todo: check why there could no vlaue for (obj_s, obj_e)
                if (obj_s, obj_e) not in ent_type:
                    print("Warning: No entity type can be found for an object")
                obj_type = ent_type.get((obj_s, obj_e), 0)  # @todo: check why there could no vlaue for (obj_s, obj_e)
                triples.add(
                    (sub_s, sub_e, obj_s, obj_e, rel2id[rel.label], sub_type, obj_type)
                )
                cores_table[sub_type, obj_type, rel2id[rel.label]] = 1

            max_tripes_count = config.get("max_tripes_count", 30)
            if len(triples) > max_tripes_count:
               print(f"Warning: {len(triples)} triples are found in a sentence. Reduced to ma_tripes_count: {max_tripes_count}")
               triples = sorted(list(triples), key=lambda x: x[0])
               triples = triples[:max_tripes_count]
               triples = set(triples)
            assert (
                len(triples) <= max_tripes_count
            ), f"triples count {len(triples)} > {max_tripes_count}"
            triples = list(triples) + [(-1, -1, -1, -1, -1, -1, -1)] * (
                max_tripes_count - len(triples)
            )

            sent_mask = torch.zeros(len(tokenized_tokens), dtype=torch.int16)
            sent_mask[sent_start:sent_end] = 1
            if config.use_cross_ner:
                sent_mask = torch.ones(len(tokenized_tokens), dtype=torch.int16)
                sent_mask[input_ids == tokenizer.pad_token_id] = 0

            sample["input_ids"] = input_ids
            sample["triples"] = torch.tensor(triples, dtype=torch.int16)
            sample["attention_mask"] = attention_mask
            sample["pos"] = torch.tensor(
                (sent_start, sent_end, sent.sentence_ix, sent.sentence_start, c),
                dtype=torch.int32,
            )
            sample["ent_maps"] = ent_maps if not config.use_spert else ent_maps_2d
            sample["sent_mask"] = sent_mask
            sample["span_mask"] = torch.tensor(span_mask, dtype=torch.int16)
            samples.append(sample)

    avg_length = sum(len(s["tokens"]) for s in samples) / len(samples)
    max_length = max(len(s["tokens"]) for s in samples)
    print(
        f"Extracted {len(samples)} samples from {data_range[1]-data_range[0]} documents, "
        f"with {num_ner} NER labels, {avg_length:.3f} avg input length, {max_length} max length"
    )
    print(f"Max Length: {max_len}, max NER: {max_ner}")

    return {
        "input_ids": torch.stack([s["input_ids"] for s in samples]),
        "attention_mask": torch.stack([s["attention_mask"] for s in samples]),
        "pos": torch.stack([s["pos"] for s in samples]),
        "triples": torch.stack([s["triples"] for s in samples]),
        "ent_maps": torch.stack([s["ent_maps"] for s in samples]),
        "sent_mask": torch.stack([s["sent_mask"] for s in samples]),
        "span_mask": torch.stack([s["span_mask"] for s in samples]),
    }


def get_language_map_dict():
    ace_rel_map = {
        "NA": "no relation in this sentence",
        "ART": "artifact person",
        "ORG-AFF": "organization affiliation",
        "GEN-AFF": "general affiliation",
        "PHYS": "physical location",
        "PER-SOC": "personal social",
        "PART-WHOLE": "part whole",
        "OTHER-AFF": "organization or general affiliation",
        "GPE-AFF": "geopolitical affiliation",
        "EMP-ORG": "employment organization",
        "DISC": "discourse",
        "USED-FOR": "used for",
        "FEATURE-OF": "feature of",
        "HYPONYM-OF": "hyponym of",
        "PART-OF": "part of",
        "COMPARE": "compare",
        "CONJUNCTION": "conjunction",
        "EVALUATE-FOR": "evaluate for",
    }
    ace_ent_map = {
        "NA": "none",
        "FAC": "entity facility",
        "WEA": "entity weapon",
        "LOC": "entity location",
        "VEH": "entity vehicle",
        "GPE": "entity geopolitical",
        "ORG": "entity organization",
        "PER": "entity person",
        "Task": "task",
        "Method": "method",
        "Metric": "metric",
        "Material": "material",
        "OtherScientificTerm": "other scientific term",
        "Generic": "generic",
    }
    tag_map = {
        "[SS]": "subject start",
        "[SE]": "subject end",
        "[OS]": "object start",
        "[OE]": "object end",
        "[ES]": "entity start",
        "[EE]": "entity end",
    }
    return ace_rel_map, ace_ent_map, tag_map


def find_subarray_position(s, a):
    """Return all start positions where sub-list a occurs in list s."""
    n, m = len(s), len(a)
    if m > n:
        return []
    return [i for i in range(n - m + 1) if s[i : i + m] == a]
