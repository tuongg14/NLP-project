import time
import random
from pathlib import Path
from math import inf

import torch
import torch.nn as nn
from tqdm import tqdm

from data import (
    train_loader, val_loader, test_loader,
    tgt_vocab, src_vocab,
    PAD_TOKEN, SOS_TOKEN, EOS_TOKEN
)
from model import Seq2Seq, Encoder, Decoder
from config import DEVICE, EMB_DIM, HID_DIM, N_LAYERS, DROPOUT, LEARNING_RATE, NUM_EPOCHS


# ===== Loss & Optimizer =====
PAD_IDX = tgt_vocab.stoi[PAD_TOKEN]
criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

enc = Encoder(len(src_vocab), EMB_DIM, HID_DIM, n_layers=N_LAYERS, dropout=DROPOUT)
dec = Decoder(len(tgt_vocab), EMB_DIM, HID_DIM, n_layers=N_LAYERS, dropout=DROPOUT)
model = Seq2Seq(enc, dec, DEVICE).to(DEVICE)

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)


# ====================================================
# ================ TRAINING FUNCTIONS =================
# ====================================================
def train_epoch(model, dataloader, optimizer, criterion, teacher_forcing_ratio=0.5, clip=1.0):
    model.train()
    epoch_loss = 0

    for src, src_lens, tgt_in, tgt_out in tqdm(dataloader, desc="train"):
        src, src_lens = src.to(DEVICE), src_lens.to(DEVICE)
        tgt_in, tgt_out = tgt_in.to(DEVICE), tgt_out.to(DEVICE)

        optimizer.zero_grad()

        output = model(src, src_lens, tgt_in, teacher_forcing_ratio=teacher_forcing_ratio)
        # output: [B, T, V]

        output_dim = output.size(-1)
        output = output[:, 1:, :].contiguous()
        tgt_out = tgt_out[:, :output.size(1)].contiguous()

        loss = criterion(output.view(-1, output_dim), tgt_out.view(-1))
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(dataloader)


# ====================================================
# =============== BLEU EVALUATION ====================
# ====================================================
import sacrebleu


def safe_tok_from_idx(vocab, idx):
    try:
        if idx < len(vocab.itos):
            tok = vocab.itos[idx]
        else:
            tok = "<unk>"
    except:
        tok = "<unk>"
    return tok


def generate_hyps_from_loader(model, dataloader, tgt_vocab, max_len=50):
    model.eval()
    hyps = []
    refs = []

    with torch.no_grad():
        for src, src_lens, tgt_in, tgt_out in tqdm(dataloader, desc="generate"):
            src, src_lens = src.to(DEVICE), src_lens.to(DEVICE)

            preds = model.greedy_decode(
                src, src_lens, max_len=max_len,
                sos_idx=tgt_vocab.stoi[SOS_TOKEN],
                eos_idx=tgt_vocab.stoi[EOS_TOKEN]
            )

            # convert predictions
            for seq in preds:
                tokens = []
                for idx in seq:
                    if idx == tgt_vocab.stoi[EOS_TOKEN]:
                        break
                    tok = safe_tok_from_idx(tgt_vocab, idx)
                    if tok not in (PAD_TOKEN, SOS_TOKEN, EOS_TOKEN):
                        tokens.append(tok)
                hyps.append(" ".join(tokens))

            # convert references
            tgt_out_np = tgt_out.cpu().numpy()
            for line in tgt_out_np:
                tokens = []
                for idx in line:
                    if idx == tgt_vocab.stoi[EOS_TOKEN]:
                        break
                    tok = safe_tok_from_idx(tgt_vocab, idx)
                    if tok not in (PAD_TOKEN, SOS_TOKEN, EOS_TOKEN):
                        tokens.append(tok)
                refs.append(" ".join(tokens))

    return hyps, refs


def compute_corpus_bleu(hyps, refs):
    bleu = sacrebleu.corpus_bleu(hyps, [refs])
    return bleu.score


# ====================================================
# ====================== TRAIN ========================
# ====================================================
def train():
    print("Training model...")

    best_val_bleu = -1
    save_dir = Path("checkpoints")
    save_dir.mkdir(exist_ok=True, parents=True)
    save_path = save_dir / "best_model.pth"

    for epoch in range(1, NUM_EPOCHS + 1):
        t0 = time.time()

        train_loss = train_epoch(
            model, train_loader, optimizer,
            criterion, teacher_forcing_ratio=0.5
        )

        # VALIDATE USING BLEU
        hyps, refs = generate_hyps_from_loader(model, val_loader, tgt_vocab)
        val_bleu = compute_corpus_bleu(hyps, refs)

        elapsed = time.time() - t0

        print(f"Epoch {epoch:03d} | Train Loss: {train_loss:.3f} | "
              f"Val BLEU: {val_bleu:.2f} | Time: {elapsed:.2f}s")

        # Save best
        if val_bleu > best_val_bleu:
            best_val_bleu = val_bleu
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "src_itos": src_vocab.itos,
                "tgt_itos": tgt_vocab.itos,
                "config": {
                    "emb_dim": EMB_DIM,
                    "hid_dim": HID_DIM,
                    "n_layers": N_LAYERS
                }
            }, save_path)
            print("Checkpoint saved.")

    print(f"\nTraining finished. Best BLEU = {best_val_bleu:.2f}")


if __name__ == "__main__":
    train()
