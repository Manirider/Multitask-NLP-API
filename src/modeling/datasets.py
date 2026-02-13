import json
import logging

import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class SentimentDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length):
        self.data = self._load_data(data_path)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def _load_data(self, path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        encoding = self.tokenizer(
            item["text"],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": torch.tensor(item["label"], dtype=torch.long),
        }


class NERDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length):
        self.data = self._load_data(data_path)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def _load_data(self, path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        tokens = item["tokens"]
        tags = item["tags"]

        tokenized_inputs = self.tokenizer(
            tokens,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            is_split_into_words=True,
            return_tensors="pt",
        )

        word_ids = tokenized_inputs.word_ids()
        previous_word_idx = None
        label_ids = []

        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(tags[word_idx])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx

        return {
            "input_ids": tokenized_inputs["input_ids"].squeeze(),
            "attention_mask": tokenized_inputs["attention_mask"].squeeze(),
            "labels": torch.tensor(label_ids, dtype=torch.long),
        }


class QADataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length):
        self.data = self._load_data(data_path)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def _load_data(self, path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        context = item["context"]
        question = item["question"]
        answers = item["answers"]

        tokenized_inputs = self.tokenizer(
            question,
            context,
            truncation="only_second",
            padding="max_length",
            max_length=self.max_length,
            return_offsets_mapping=True,
            return_tensors="pt",
        )

        offset_mapping = tokenized_inputs.pop("offset_mapping").squeeze().tolist()
        input_ids = tokenized_inputs["input_ids"].squeeze()
        attention_mask = tokenized_inputs["attention_mask"].squeeze()

        if not answers["answer_start"]:
            start_position = 0
            end_position = 0
        else:
            start_char = answers["answer_start"][0]
            end_char = start_char + len(answers["text"][0])
            sequence_ids = tokenized_inputs.sequence_ids()

            idx_ptr = 0
            while sequence_ids[idx_ptr] != 1:
                idx_ptr += 1
            context_start = idx_ptr
            while idx_ptr < len(sequence_ids) and sequence_ids[idx_ptr] == 1:
                idx_ptr += 1
            context_end = idx_ptr - 1

            if (
                offset_mapping[context_start][0] > start_char
                or offset_mapping[context_end][1] < end_char
            ):
                start_position = 0
                end_position = 0
            else:
                idx_ptr = context_start
                while (
                    idx_ptr <= context_end and offset_mapping[idx_ptr][0] <= start_char
                ):
                    idx_ptr += 1
                start_position = idx_ptr - 1

                idx_ptr = context_end
                while (
                    idx_ptr >= context_start and offset_mapping[idx_ptr][1] >= end_char
                ):
                    idx_ptr -= 1
                end_position = idx_ptr + 1

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "start_positions": torch.tensor(start_position, dtype=torch.long),
            "end_positions": torch.tensor(end_position, dtype=torch.long),
        }
