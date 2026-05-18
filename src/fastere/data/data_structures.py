"""
Data structures for document-level entity and relation data.
Based on DYGIE++ codebase.
"""

import copy
import json
from collections import Counter

import numpy as np
import torch
from torch.utils.data import TensorDataset


def fields_to_batches(d, keys_to_ignore=[]):
    keys = [key for key in d.keys() if key not in keys_to_ignore]
    lengths = [len(d[k]) for k in keys]
    assert len(set(lengths)) == 1
    length = lengths[0]
    return [{k: d[k][i] for k in keys} for i in range(length)]


def get_sentence_of_span(span, sentence_starts, doc_tokens):
    sentence_ends = [x - 1 for x in sentence_starts[1:]] + [doc_tokens - 1]
    in_between = [
        span[0] >= start and span[1] <= end
        for start, end in zip(sentence_starts, sentence_ends)
    ]
    assert sum(in_between) == 1
    return in_between.index(True)


class Dataset:
    def __init__(self, json_file, pred_file=None, doc_range=None):
        self.js = self._read(json_file, pred_file)
        if doc_range is not None:
            self.js = self.js[doc_range[0] : doc_range[1]]
        self.documents = [Document(js) for js in self.js]

    def update_from_js(self, js):
        self.js = js
        self.documents = [Document(js) for js in self.js]

    def _read(self, json_file, pred_file=None):
        gold_docs = [json.loads(line) for line in open(json_file)]
        if pred_file is None:
            return gold_docs
        pred_docs = [json.loads(line) for line in open(pred_file)]
        merged_docs = []
        for gold, pred in zip(gold_docs, pred_docs):
            assert gold["doc_key"] == pred["doc_key"]
            assert gold["sentences"] == pred["sentences"]
            merged = copy.deepcopy(gold)
            for k, v in pred.items():
                if "predicted" in k:
                    merged[k] = v
            merged_docs.append(merged)
        return merged_docs

    def __getitem__(self, ix):
        return self.documents[ix]

    def __len__(self):
        return len(self.documents)


class Document:
    def __init__(self, js):
        self._doc_key = js["doc_key"]
        entries = fields_to_batches(
            js, ["doc_key", "clusters", "predicted_clusters", "section_starts"]
        )
        sentence_lengths = [len(entry["sentences"]) for entry in entries]
        sentence_starts = np.cumsum(sentence_lengths)
        sentence_starts = np.roll(sentence_starts, 1)
        sentence_starts[0] = 0
        self.sentence_starts = sentence_starts
        self.sentences = [
            Sentence(entry, sentence_start, sentence_ix)
            for sentence_ix, (entry, sentence_start) in enumerate(
                zip(entries, sentence_starts)
            )
        ]
        if "clusters" in js:
            self.clusters = [
                Cluster(entry, i, self) for i, entry in enumerate(js["clusters"])
            ]
        if "predicted_clusters" in js:
            self.predicted_clusters = [
                Cluster(entry, i, self)
                for i, entry in enumerate(js["predicted_clusters"])
            ]

    def __repr__(self):
        return "\n".join(
            [
                str(i) + ": " + " ".join(sent.text)
                for i, sent in enumerate(self.sentences)
            ]
        )

    def __getitem__(self, ix):
        return self.sentences[ix]

    def __len__(self):
        return len(self.sentences)

    def print_plaintext(self):
        for sent in self:
            print(" ".join(sent.text))

    def find_cluster(self, entity, predicted=True):
        clusters = self.predicted_clusters if predicted else self.clusters
        for clust in clusters:
            for entry in clust:
                if entry.span == entity.span:
                    return clust
        return None

    @property
    def n_tokens(self):
        return sum([len(sent) for sent in self.sentences])


class Sentence:
    def __init__(self, entry, sentence_start, sentence_ix):
        self.sentence_start = sentence_start
        self.text = entry["sentences"]
        self.sentence_ix = sentence_ix
        if "ner_flavor" in entry:
            self.ner = [
                NER(this_ner, self.text, sentence_start, flavor=this_flavor)
                for this_ner, this_flavor in zip(entry["ner"], entry["ner_flavor"])
            ]
        elif "ner" in entry:
            self.ner = [
                NER(this_ner, self.text, sentence_start) for this_ner in entry["ner"]
            ]
        if "relations" in entry:
            self.relations = [
                Relation(this_relation, self.text, sentence_start)
                for this_relation in entry["relations"]
            ]
        if "events" in entry:
            self.events = Events(entry["events"], self.text, sentence_start)
        if "predicted_ner" in entry:
            self.predicted_ner = [
                NER(this_ner, self.text, sentence_start, flavor=None)
                for this_ner in entry["predicted_ner"]
            ]
        if "predicted_relations" in entry:
            self.predicted_relations = [
                Relation(this_relation, self.text, sentence_start)
                for this_relation in entry["predicted_relations"]
            ]
        if "predicted_events" in entry:
            self.predicted_events = Events(
                entry["predicted_events"], self.text, sentence_start
            )
        if "top_spans" in entry:
            self.top_spans = [
                NER(this_ner, self.text, sentence_start, flavor=None)
                for this_ner in entry["top_spans"]
            ]

    def __repr__(self):
        the_text = " ".join(self.text)
        the_lengths = np.array([len(x) for x in self.text])
        tok_ixs = ""
        for i, offset in enumerate(the_lengths):
            true_offset = offset if i < 10 else offset - 1
            tok_ixs += str(i)
            tok_ixs += " " * true_offset
        return the_text + "\n" + tok_ixs

    def __len__(self):
        return len(self.text)

    def get_flavor(self, argument):
        the_ner = [x for x in self.ner if x.span == argument.span]
        return the_ner[0].flavor if the_ner else None


class Span:
    def __init__(self, start, end, text, sentence_start):
        self.start_doc = start
        self.end_doc = end
        self.span_doc = (self.start_doc, self.end_doc)
        self.start_sent = start - sentence_start
        self.end_sent = end - sentence_start
        self.span_sent = (self.start_sent, self.end_sent)
        self.text = text[self.start_sent : self.end_sent + 1]

    def __repr__(self):
        return str((self.start_sent, self.end_sent, self.text))

    def __eq__(self, other):
        return (
            self.span_doc == other.span_doc
            and self.span_sent == other.span_sent
            and self.text == other.text
        )

    def __hash__(self):
        return hash(self.span_doc + self.span_sent + (" ".join(self.text),))


class Token:
    def __init__(self, ix, text, sentence_start):
        self.ix_doc = ix
        self.ix_sent = ix - sentence_start
        self.text = text[self.ix_sent]

    def __repr__(self):
        return str((self.ix_sent, self.text))


class Trigger:
    def __init__(self, token, label):
        self.token = token
        self.label = label

    def __repr__(self):
        return self.token.__repr__()[:-1] + ", " + self.label + ")"


class Argument:
    def __init__(self, span, role, event_type):
        self.span = span
        self.role = role
        self.event_type = event_type

    def __repr__(self):
        return (
            self.span.__repr__()[:-1] + ", " + self.event_type + ", " + self.role + ")"
        )

    def __eq__(self, other):
        return (
            self.span == other.span
            and self.role == other.role
            and self.event_type == other.event_type
        )

    def __hash__(self):
        return self.span.__hash__() + hash((self.role, self.event_type))


class NER:
    def __init__(self, ner, text, sentence_start, flavor=None):
        self.span = Span(ner[0], ner[1], text, sentence_start)
        self.label = ner[2]
        self.flavor = flavor

    def __repr__(self):
        return self.span.__repr__() + ": " + self.label

    def __eq__(self, other):
        return (
            self.span == other.span
            and self.label == other.label
            and self.flavor == other.flavor
        )


class Relation:
    def __init__(self, relation, text, sentence_start):
        start1, end1, start2, end2, label = (
            relation[0],
            relation[1],
            relation[2],
            relation[3],
            relation[4],
        )
        self.pair = (
            Span(start1, end1, text, sentence_start),
            Span(start2, end2, text, sentence_start),
        )
        self.label = label

    def __repr__(self):
        return (
            self.pair[0].__repr__() + ", " + self.pair[1].__repr__() + ": " + self.label
        )

    def __eq__(self, other):
        return self.pair == other.pair and self.label == other.label


class AtomicRelation:
    def __init__(self, ent0, ent1, label):
        self.ent0 = ent0
        self.ent1 = ent1
        self.label = label

    @classmethod
    def from_relation(cls, relation):
        return cls(
            " ".join(relation.pair[0].text),
            " ".join(relation.pair[1].text),
            relation.label,
        )

    def __repr__(self):
        return f"({self.ent0} | {self.ent1} | {self.label})"


class Event:
    def __init__(self, event, text, sentence_start):
        trig = event[0]
        args = event[1:]
        trigger_token = Token(trig[0], text, sentence_start)
        self.trigger = Trigger(trigger_token, trig[1])
        self.arguments = [
            Argument(
                Span(arg[0], arg[1], text, sentence_start), arg[2], self.trigger.label
            )
            for arg in args
        ]

    def __repr__(self):
        res = "<" + self.trigger.__repr__() + ":\n"
        for arg in self.arguments:
            res += 6 * " " + arg.__repr__() + ";\n"
        return res[:-2] + ">"


class Events:
    def __init__(self, events_json, text, sentence_start):
        self.event_list = [Event(e, text, sentence_start) for e in events_json]
        self.triggers = set(event.trigger for event in self.event_list)
        self.arguments = set(
            arg for event in self.event_list for arg in event.arguments
        )

    def __len__(self):
        return len(self.event_list)

    def __getitem__(self, i):
        return self.event_list[i]

    def __repr__(self):
        return "\n\n".join(event.__repr__() for event in self.event_list)

    def span_matches(self, argument):
        return {
            c for c in self.arguments if c.span.span_sent == argument.span.span_sent
        }

    def event_type_matches(self, argument):
        return {
            c
            for c in self.span_matches(argument)
            if c.event_type == argument.event_type
        }

    def matches_except_event_type(self, argument):
        return {
            c
            for c in self.span_matches(argument)
            if c.event_type != argument.event_type and c.role == argument.role
        }

    def exact_match(self, argument):
        return any(c == argument for c in self.arguments)


class Cluster:
    def __init__(self, cluster, cluster_id, document):
        members = []
        for entry in cluster:
            sentence_ix = get_sentence_of_span(
                entry, document.sentence_starts, document.n_tokens
            )
            sentence = document[sentence_ix]
            span = Span(entry[0], entry[1], sentence.text, sentence.sentence_start)
            ners = [x for x in sentence.ner if x.span == span]
            assert len(ners) <= 1
            members.append(
                ClusterMember(span, ners[0] if ners else None, sentence, cluster_id)
            )
        self.members = members
        self.cluster_id = cluster_id

    def __repr__(self):
        return f"{self.cluster_id}: " + self.members.__repr__()

    def __getitem__(self, ix):
        return self.members[ix]


class ClusterMember:
    def __init__(self, span, ner, sentence, cluster_id):
        self.span = span
        self.ner = ner
        self.sentence = sentence
        self.cluster_id = cluster_id

    def __repr__(self):
        return f"<{self.sentence.sentence_ix}> " + self.span.__repr__()


# ---------------------------------------------------------------------------
# Evaluation utilities
# ---------------------------------------------------------------------------


def safe_div(num, denom):
    return num / denom if denom > 0 else 0


def compute_f1(predicted, gold, matched):
    precision = safe_div(matched, predicted)
    recall = safe_div(matched, gold)
    f1 = safe_div(2 * precision * recall, precision + recall)
    return dict(precision=precision, recall=recall, f1=f1)


def evaluate_sent(sent, counts):
    correct_ner = set()
    counts["ner_gold"] += len(sent.ner)
    counts["ner_predicted"] += len(sent.predicted_ner)
    for prediction in sent.predicted_ner:
        if any(prediction == actual for actual in sent.ner):
            counts["ner_matched"] += 1
            correct_ner.add(prediction.span)

    counts["relations_gold"] += len(sent.relations)
    counts["relations_predicted"] += len(sent.predicted_relations)
    for prediction in sent.predicted_relations:
        if any(prediction == actual for actual in sent.relations):
            counts["relations_matched"] += 1
            if prediction.pair[0] in correct_ner and prediction.pair[1] in correct_ner:
                counts["strict_relations_matched"] += 1
    return counts


def evaluate_predictions(dataset):
    counts = Counter()
    for doc in dataset:
        for sent in doc:
            counts = evaluate_sent(sent, counts)
    return dict(
        ner=compute_f1(
            counts["ner_predicted"], counts["ner_gold"], counts["ner_matched"]
        ),
        relation=compute_f1(
            counts["relations_predicted"],
            counts["relations_gold"],
            counts["relations_matched"],
        ),
        strict_relation=compute_f1(
            counts["relations_predicted"],
            counts["relations_gold"],
            counts["strict_relations_matched"],
        ),
    )
