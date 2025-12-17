# src/data.py
"""
Data loading, tokenization, vocabulary building, Dataset class, and collate_fn
for the English–French Machine Translation (Seq2Seq LSTM) project.
"""

import os
from collections import Counter
from pathlib import Path

import spacy
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


# ---------------------------------------------------------------------
# 1. Load spaCy tokenizers
# ---------------------------------------------------------------------

try:
    SPACY_EN = spacy.load("en_core_web_sm")
    SPACY_FR = spacy.load("fr_core_news_sm")
except Exception as e:
    raise RuntimeError(
        "❌ spaCy models not found. Please install:\n"
        "python -m spacy download en_core_web_sm\n"
        "python -m spacy download fr_core_news_sm"
    )


# Tokenization functions
def tokenize_en(text):
    return [tok.text.lower() for tok in SPACY_EN.tokenizer(text)]


def tokenize_fr(text):
    return [tok.text.lower() for tok in SPACY_FR.tokenizer(text)]


# ---------------------------------------------------------------------
# 2. Read text files
# ---------------------------------------------------------------------

def read_lines(path):
    """Read a file and strip blank lines."""
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


# ---------------------------------------------------------------------
# 3. Vocab class
# ---------------------------------------------------------------------

PAD_TOKEN = "<pad>"
SOS_TOKEN = "<sos>"
EOS_TOKEN = "<eos>"
UNK_TOKEN = "<unk>"


class Vocab:
    """
    Simple vocabulary class:
    - builds from token lists
    - numericalize() converts tokens to IDs
    """
    def __init__(self, tokens_list, min_freq=2, max_size=None):
        counter = Counter()
        for toks in tokens_list:
            counter.update(toks)

        # Special tokens first
        self.itos = [PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN]

        # Add vocab tokens
        for tok, freq in counter.most_common():
            if freq < min_freq:
                continue
            if tok in self.itos:
                continue
            self.itos.append(tok)
            if max_size and len(self.itos) >= max_size:
                break

        self.stoi = {tok: i for i, tok in enumerate(self.itos)}

    def __len__(self):
        return len(self.itos)

    def numericalize(self, tokens):
        """Convert list of tokens → list of indices."""
        return [self.stoi.get(t, self.stoi[UNK_TOKEN]) for t in tokens]


# ---------------------------------------------------------------------
# 4. Dataset class
# ---------------------------------------------------------------------

class TranslationDataset(Dataset):
    """
    Stores:
    - src token list: [["a","man","is"...], ...]
    - tgt token list: [["un","homme","..."], ...]

    Returns:
    - src_ids
    - tgt_in_ids  (with <sos>)
    - tgt_out_ids (with <eos>)
    """
    def __init__(self, src_tok_list, tgt_tok_list, src_vocab, tgt_vocab, max_len=100):
        assert len(src_tok_list) == len(tgt_tok_list)

        self.src = src_tok_list
        self.tgt = tgt_tok_list
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.src)

    def __getitem__(self, idx):
        src_tokens = self.src[idx][:self.max_len]
        tgt_tokens = self.tgt[idx][:self.max_len - 2]  # allow space for SOS/EOS

        src_ids = self.src_vocab.numericalize(src_tokens)

        # Decoder input starts with <sos>
        tgt_ids_in = [self.tgt_vocab.stoi[SOS_TOKEN]] + \
            self.tgt_vocab.numericalize(tgt_tokens)

        # Decoder output ends with <eos>
        tgt_ids_out = self.tgt_vocab.numericalize(tgt_tokens) + \
            [self.tgt_vocab.stoi[EOS_TOKEN]]

        return (
            torch.tensor(src_ids, dtype=torch.long),
            torch.tensor(tgt_ids_in, dtype=torch.long),
            torch.tensor(tgt_ids_out, dtype=torch.long),
        )


# ---------------------------------------------------------------------
# 5. Collate function for DataLoader
# ---------------------------------------------------------------------

def build_collate_fn(src_vocab, tgt_vocab):
    """
    Returns a collate_fn with correct PAD idx preloaded.

    Output:
    - src_pad: (B, src_len)
    - src_lens: (B,)
    - tgt_in_pad: (B, tgt_len)
    - tgt_out_pad: (B, tgt_len)
    """

    def collate_fn(batch):
        src_batch, tgt_in_batch, tgt_out_batch = zip(*batch)

        src_lens = torch.tensor([len(x) for x in src_batch], dtype=torch.long)

        src_pad = pad_sequence(
            src_batch,
            batch_first=True,
            padding_value=src_vocab.stoi[PAD_TOKEN],
        )
        tgt_in_pad = pad_sequence(
            tgt_in_batch,
            batch_first=True,
            padding_value=tgt_vocab.stoi[PAD_TOKEN],
        )
        tgt_out_pad = pad_sequence(
            tgt_out_batch,
            batch_first=True,
            padding_value=tgt_vocab.stoi[PAD_TOKEN],
        )

        return src_pad, src_lens, tgt_in_pad, tgt_out_pad

    return collate_fn


# ---------------------------------------------------------------------
# 6. Helper: build vocab from tokenized dataset
# ---------------------------------------------------------------------

def build_vocab_from_token_lists(token_lists, min_freq=2, max_size=10000):
    return Vocab(token_lists, min_freq=min_freq, max_size=max_size)
