import os, random, re, string
from collections import Counter
from tqdm import tqdm
import pickle

from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

import nltk

nltk.download("punkt", quiet=True)
from transformers import T5TokenizerFast
import torch

PAD_IDX = 0


class T5Dataset(Dataset):
    def __init__(self, data_folder, split):
        self.data_folder = data_folder
        self.split = split

        # Initialize T5 tokenizer
        self.tokenizer = T5TokenizerFast.from_pretrained("google-t5/t5-small")

        # Process and load the data
        self.encoder_input_ids, self.decoder_input_ids, self.decoder_target_ids = (
            self.process_data(data_folder, split, self.tokenizer)
        )

    def process_data(self, data_folder, split, tokenizer):
        import os

        # Load natural language queries
        nl_path = os.path.join('data', f'{split}.nl')
        with open(nl_path, "r") as f:
            nl_queries = [line.strip() for line in f.readlines()]

        # For test set, we don't have SQL queries
        if split == "test":
            encoder_input_ids = []
            for query in nl_queries:
                input_text = f"translate to SQL: {query}"
                encoded = tokenizer(
                    input_text, add_special_tokens=True, truncation=True, max_length=512
                )
                encoder_input_ids.append(encoded["input_ids"])

            return encoder_input_ids, None, None

        # For train/dev, load SQL queries
        sql_path = os.path.join('data', f'{split}.sql')
        with open(sql_path, "r") as f:
            sql_queries = [line.strip() for line in f.readlines()]

        assert len(nl_queries) == len(sql_queries)

        encoder_input_ids = []
        decoder_input_ids = []
        decoder_target_ids = []

        for nl_query, sql_query in zip(nl_queries, sql_queries):
            # Tokenize encoder input
            input_text = f"translate to SQL: {nl_query}"
            enc_encoded = tokenizer(
                input_text, add_special_tokens=True, truncation=True, max_length=512
            )
            encoder_input_ids.append(enc_encoded["input_ids"])

            # Tokenize decoder
            dec_encoded = tokenizer(
                sql_query, add_special_tokens=True, truncation=True, max_length=512
            )

            # Decoder input: prepend with pad token
            decoder_input = [tokenizer.pad_token_id] + dec_encoded["input_ids"][:-1]
            decoder_input_ids.append(decoder_input)

            # Decoder targets
            decoder_target_ids.append(dec_encoded["input_ids"])

        return encoder_input_ids, decoder_input_ids, decoder_target_ids

    def __len__(self):
        return len(self.encoder_input_ids)

    def __getitem__(self, idx):
        if self.split == "test":
            return {
                "encoder_input_ids": self.encoder_input_ids[idx],
            }
        else:
            return {
                "encoder_input_ids": self.encoder_input_ids[idx],
                "decoder_input_ids": self.decoder_input_ids[idx],
                "decoder_target_ids": self.decoder_target_ids[idx],
            }


def normal_collate_fn(batch):
    encoder_input_ids = [torch.tensor(item["encoder_input_ids"]) for item in batch]
    decoder_input_ids = [torch.tensor(item["decoder_input_ids"]) for item in batch]
    decoder_target_ids = [torch.tensor(item["decoder_target_ids"]) for item in batch]

    encoder_ids = pad_sequence(
        encoder_input_ids, batch_first=True, padding_value=PAD_IDX
    )
    decoder_inputs = pad_sequence(
        decoder_input_ids, batch_first=True, padding_value=PAD_IDX
    )
    decoder_targets = pad_sequence(
        decoder_target_ids, batch_first=True, padding_value=PAD_IDX
    )

    encoder_mask = (encoder_ids != PAD_IDX).long()
    initial_decoder_inputs = torch.full((len(batch), 1), PAD_IDX, dtype=torch.long)

    return (
        encoder_ids,
        encoder_mask,
        decoder_inputs,
        decoder_targets,
        initial_decoder_inputs,
    )


def test_collate_fn(batch):
    encoder_input_ids = [torch.tensor(item["encoder_input_ids"]) for item in batch]
    encoder_ids = pad_sequence(
        encoder_input_ids, batch_first=True, padding_value=PAD_IDX
    )
    encoder_mask = (encoder_ids != PAD_IDX).long()
    initial_decoder_inputs = torch.full((len(batch), 1), PAD_IDX, dtype=torch.long)

    return encoder_ids, encoder_mask, initial_decoder_inputs


def get_dataloader(batch_size, split):
    data_folder = "."
    dset = T5Dataset(data_folder, split)
    shuffle = split == "train"
    collate_fn = normal_collate_fn if split != "test" else test_collate_fn

    dataloader = DataLoader(
        dset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn
    )
    return dataloader


def load_t5_data(batch_size, test_batch_size):
    train_loader = get_dataloader(batch_size, "train")
    dev_loader = get_dataloader(test_batch_size, "dev")
    test_loader = get_dataloader(test_batch_size, "test")

    return train_loader, dev_loader, test_loader


def load_lines(path):
    with open(path, "r") as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
    return lines


def load_prompting_data(data_folder):
    train_x = load_lines(os.path.join(data_folder, "train.nl"))
    train_y = load_lines(os.path.join(data_folder, "train.sql"))
    dev_x = load_lines(os.path.join(data_folder, "dev.nl"))
    dev_y = load_lines(os.path.join(data_folder, "dev.sql"))
    test_x = load_lines(os.path.join(data_folder, "test.nl"))

    return train_x, train_y, dev_x, dev_y, test_x
