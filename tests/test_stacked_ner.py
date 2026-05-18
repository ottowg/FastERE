"""Tests for _assign_ner_to_layers (stacked BIO NER preprocessing helper)."""

import pytest

from fastere.data.utils import _assign_ner_to_layers


class MockSpan:
    def __init__(self, start, end):
        self.start_sent = start
        self.end_sent = end


class MockNER:
    def __init__(self, start, end, label):
        self.span = MockSpan(start, end)
        self.label = label


class TestAssignNerToLayers:
    def test_non_overlapping_all_layer_zero(self):
        entities = [
            MockNER(0, 1, "PER"),
            MockNER(3, 4, "ORG"),
            MockNER(6, 7, "LOC"),
        ]
        assignments = _assign_ner_to_layers(entities, num_layers=2, label_preference=[])
        assert len(assignments) == 3
        assert all(layer_idx == 0 for _, layer_idx in assignments)

    def test_nested_outer_layer_zero_inner_layer_one(self):
        outer = MockNER(1, 3, "MLModel")
        inner = MockNER(1, 1, "Dataset")
        assignments = _assign_ner_to_layers(
            [outer, inner], num_layers=2, label_preference=[]
        )
        assert len(assignments) == 2
        by_ner = {id(ner): layer_idx for ner, layer_idx in assignments}
        assert by_ner[id(outer)] == 0
        assert by_ner[id(inner)] == 1

    def test_same_span_with_preference(self):
        ner_a = MockNER(2, 4, "ORG")
        ner_b = MockNER(2, 4, "PER")
        assignments = _assign_ner_to_layers(
            [ner_a, ner_b], num_layers=2, label_preference=["PER", "ORG"]
        )
        assert len(assignments) == 2
        by_ner = {id(ner): layer_idx for ner, layer_idx in assignments}
        assert by_ner[id(ner_b)] == 0, "PER is preferred, should go to layer 0"
        assert by_ner[id(ner_a)] == 1

    def test_same_span_without_preference_lexical_deterministic(self):
        ner_x = MockNER(0, 2, "Zebra")
        ner_y = MockNER(0, 2, "Apple")
        assignments1 = _assign_ner_to_layers(
            [ner_x, ner_y], num_layers=2, label_preference=[]
        )
        assignments2 = _assign_ner_to_layers(
            [ner_y, ner_x], num_layers=2, label_preference=[]
        )
        layers1 = {ner.label: layer_idx for ner, layer_idx in assignments1}
        layers2 = {ner.label: layer_idx for ner, layer_idx in assignments2}
        assert (
            layers1 == layers2
        ), "Lexical tiebreak must be deterministic regardless of input order"
        assert layers1["Apple"] == 0
        assert layers1["Zebra"] == 1

    def test_insufficient_layers_drops_entity(self, capsys):
        ner_a = MockNER(0, 2, "PER")
        ner_b = MockNER(0, 2, "ORG")
        ner_c = MockNER(0, 2, "LOC")
        assignments = _assign_ner_to_layers(
            [ner_a, ner_b, ner_c], num_layers=2, label_preference=[]
        )
        assert len(assignments) == 2
        captured = capsys.readouterr()
        assert "dropped" in captured.out
        assigned_labels = {ner.label for ner, _ in assignments}
        # Lexical order: LOC < ORG < PER — so LOC→layer0, ORG→layer1, PER is dropped.
        assert "PER" not in assigned_labels
