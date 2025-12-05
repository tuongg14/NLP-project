import argparse
from functools import partial
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm


from src.config_loader import load_config, get_checkpoint_path
from src.data import load_datasets_and_vocabs, collate_fn
from src.model import Encoder, Decoder, Seq2Seq
from src.utils import set_seed, save_checkpoint

def train_loop(model, dataloader, optimizer, criterion, config):
    model.train()
    epoch_loss = 0
    for src_padded, src_lengths, tgt_padded, tgt_lengths in tqdm(dataloader):
        optimizer.zero_grad()
        outputs = model(src_padded, src_lengths, tgt_padded, teacher_forcing_ratio=config['train']['teacher_forcing_ratio'])
        logits = outputs[:, 1:, :].contiguous()
        target = tgt_padded[:, 1:].contiguous()
        loss = criterion(logits.view(-1, logits.size(-1)), target.view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config['train'].get('clip_grad_norm', 1.0))
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss / len(dataloader)