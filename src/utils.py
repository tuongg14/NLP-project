# utils.py
import torch
import random
import numpy as np
import spacy

# special tokens
PAD_TOKEN = "<pad>"
SOS_TOKEN = "<sos>"
EOS_TOKEN = "<eos>"
UNK_TOKEN = "<unk>"

# load spacy only once
spacy_en = spacy.load("en_core_web_sm")
spacy_fr = spacy.load("fr_core_news_sm")


def tokenize_en(text):
    return [tok.text.lower() for tok in spacy_en.tokenizer(text)]


def tokenize_fr(text):
    return [tok.text.lower() for tok in spacy_fr.tokenizer(text)]


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def pad_sequence(seq_ids, max_len, pad_idx):
    """
    Pad/cut sequences to a fixed length.
    seq_ids: list[int]
    """
    if len(seq_ids) >= max_len:
        return seq_ids[:max_len]
    return seq_ids + [pad_idx] * (max_len - len(seq_ids))


def collate_fn(batch):
    """
    batch = list of (src_ids, tgt_ids)
    Converts to padded tensors + lengths.
    """
    src_batch, tgt_batch = zip(*batch)

    src_lens = [len(x) for x in src_batch]
    tgt_lens = [len(x) for x in tgt_batch]

    max_src = max(src_lens)
    max_tgt = max(tgt_lens)

    PAD = src_batch[0].vocab.stoi[PAD_TOKEN] if hasattr(src_batch[0], "vocab") else 0

    src_tensor = torch.zeros(len(batch), max_src, dtype=torch.long)
    tgt_tensor = torch.zeros(len(batch), max_tgt, dtype=torch.long)

    for i, (s, t) in enumerate(zip(src_batch, tgt_batch)):
        src_tensor[i, :len(s)] = torch.tensor(s)
        tgt_tensor[i, :len(t)] = torch.tensor(t)

    return src_tensor, torch.tensor(src_lens), tgt_tensor[:, :-1], tgt_tensor[:, 1:]
