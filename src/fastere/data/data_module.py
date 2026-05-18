import os

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer

from fastere.data.data_structures import Dataset
from fastere.data.utils import convert_dataset_to_samples
from fastere.utils.cprint import cp


class DataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.model.model_name_or_path)

        self.data_train = None
        self.data_val = None
        self.data_test = None

        self.rel_num = len(config.dataset.rels)
        self.id2rel = config.dataset.rels
        self.rel2id = {name: idx for idx, name in enumerate(config.dataset.rels)}

        self.ner_num = len(config.dataset.ents)
        self.id2ner = config.dataset.ents
        self.ner2id = {name: idx for idx, name in enumerate(config.dataset.ents)}

    def setup(self, stage=None):
        if stage in ("fit", None):
            self.data_train = self._get_dataset("train")
            self.data_val = self._get_dataset("val")
        if stage in ("test", None):
            self.data_test = self._get_dataset("test")
        if stage == "predict":
            self.data_test = self._get_predict_items()

    def _get_dataset(self, mode):
        print(cp.green(f"Loading {mode} data..."), end=" ")
        name = self.config.dataset.name

        if name in ("ace2005", "ace2004", "scierc"):
            if name == "ace2004":
                piece = self.config.data_piece or 0
                assert piece in range(5), "data_piece must be in [0, 1, 2, 3, 4]"
                filename = self.config.dataset[mode][piece]
            else:
                filename = self.config.dataset[mode]

            cache_tag = f".max{self.config.max_seq_len}.ctw{self.config.context_window}"
            if self.config.use_cross_ner:
                cache_tag += ".cross_ner"

            cache_path = filename + cache_tag + ".cache"
            if os.path.exists(cache_path) and self.config.use_cache:
                print(cp.green(f"Loading from cache {cache_path}"))
                datasets = torch.load(cache_path, weights_only=False)
            else:
                datasets = self._load_ace(mode, filename)
                if self.config.use_cache:
                    torch.save(datasets, cache_path)
        else:
            raise NotImplementedError(f"Dataset '{name}' not implemented")

        return datasets

    def _load_ace(self, mode, filename):
        dataset = Dataset(filename)
        features = convert_dataset_to_samples(
            dataset, self.config, self.tokenizer, is_test=(mode == "test")
        )
        return TensorDataset(
            features["input_ids"],
            features["attention_mask"],
            features["pos"],
            features["triples"],
            features["ent_maps"],
            features["sent_mask"],
            features["span_mask"],
        )

    def _get_predict_items(self):
        dataset = Dataset(self.config.dataset["test"])
        items = []
        for doc in dataset:
            doc_text = " ".join(" ".join(sent.text) for sent in doc.sentences)
            for sent in doc.sentences:
                item = {
                    "doc": doc_text,
                    "sent": " ".join(sent.text),
                    "entities": [
                        {
                            "entity text": " ".join(ent.span.text),
                            "entity type": ent.label,
                        }
                        for ent in sent.ner
                    ],
                    "relations": [
                        {
                            "subject": " ".join(rel.pair[0].text),
                            "object": " ".join(rel.pair[1].text),
                            "relation": rel.label,
                        }
                        for rel in sent.relations
                    ],
                }
                if item["relations"]:
                    items.append(item)
        return items

    def train_dataloader(self):
        return DataLoader(
            self.data_train,
            shuffle=True,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_worker,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.data_val,
            shuffle=False,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_worker,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.data_test,
            shuffle=False,
            batch_size=self.config.test_batch_size,
            num_workers=self.config.num_worker,
            pin_memory=False,
        )
