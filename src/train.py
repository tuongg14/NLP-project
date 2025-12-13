import time
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import sacrebleu

from config_loader import load_config
from data import (
    read_lines, tokenize_en, tokenize_fr,
    build_vocab_from_token_lists, TranslationDataset, build_collate_fn,
    PAD_TOKEN, SOS_TOKEN, EOS_TOKEN,
)
from model import Encoder, Seq2SeqWithAttention, AttentionDecoder


# ====================================================
# ================ TRAINING FUNCTIONS =================
# ====================================================
def train_epoch(model, dataloader, optimizer, criterion,
                teacher_forcing_ratio=0.5, clip=1.0, device="cpu"):
    model.train()
    epoch_loss = 0.0

    for src, src_lens, tgt_in, tgt_out in tqdm(dataloader, desc="train"):
        src, src_lens = src.to(device), src_lens.to(device)
        tgt_in, tgt_out = tgt_in.to(device), tgt_out.to(device)

        optimizer.zero_grad()

        # output: [B, T, V]
        output = model(src, src_lens, tgt_in, teacher_forcing_ratio=teacher_forcing_ratio)
        vocab_size = output.size(-1)

        # bỏ timestep 0 (ứng với <sos>)
        output = output[:, 1:, :].contiguous()   # [B, T-1, V]
        tgt_out = tgt_out[:, 1:].contiguous()    # [B, T-1]

        loss = criterion(output.view(-1, vocab_size), tgt_out.view(-1))
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(dataloader)


def evaluate_epoch(model, dataloader, criterion, device="cpu"):
    model.eval()
    epoch_loss = 0.0

    with torch.no_grad():
        for src, src_lens, tgt_in, tgt_out in tqdm(dataloader, desc="val"):
            src, src_lens = src.to(device), src_lens.to(device)
            tgt_in, tgt_out = tgt_in.to(device), tgt_out.to(device)

            output = model(src, src_lens, tgt_in, teacher_forcing_ratio=0.0)
            vocab_size = output.size(-1)

            output = output[:, 1:, :].contiguous()
            tgt_out = tgt_out[:, 1:].contiguous()

            loss = criterion(output.view(-1, vocab_size), tgt_out.view(-1))
            epoch_loss += loss.item()

    return epoch_loss / len(dataloader)


# ====================================================
# ================= BLEU EVALUATION ==================
# ====================================================
def safe_tok_from_idx(vocab, idx: int):
    try:
        if 0 <= idx < len(vocab.itos):
            return vocab.itos[idx]
    except Exception:
        pass
    return "<unk>"


def generate_hyps_from_loader(model, dataloader, tgt_vocab, max_len=50, device="cpu"):
    """
    Trả về:
      hyps: list[str]
      refs: list[str]
    """
    model.eval()
    hyps, refs = [], []

    with torch.no_grad():
        for src, src_lens, tgt_in, tgt_out in tqdm(dataloader, desc="generate"):
            src, src_lens = src.to(device), src_lens.to(device)

            preds = model.greedy_decode(
                src, src_lens,
                max_len=max_len,
                sos_idx=tgt_vocab.stoi[SOS_TOKEN],
                eos_idx=tgt_vocab.stoi[EOS_TOKEN],
            )

            # hypotheses
            for seq in preds:
                tokens = []
                for idx in seq:
                    if idx == tgt_vocab.stoi[EOS_TOKEN]:
                        break
                    tok = safe_tok_from_idx(tgt_vocab, int(idx))
                    if tok not in (PAD_TOKEN, SOS_TOKEN, EOS_TOKEN):
                        tokens.append(tok)
                hyps.append(" ".join(tokens))

            # references (từ tgt_out)
            tgt_np = tgt_out.cpu().numpy()
            for line in tgt_np:
                tokens = []
                for idx in line:
                    if idx == tgt_vocab.stoi[EOS_TOKEN]:
                        break
                    tok = safe_tok_from_idx(tgt_vocab, int(idx))
                    if tok not in (PAD_TOKEN, SOS_TOKEN, EOS_TOKEN):
                        tokens.append(tok)
                refs.append(" ".join(tokens))

    return hyps, refs


def compute_corpus_bleu(hyps, refs):
    bleu = sacrebleu.corpus_bleu(hyps, [refs], force=True)
    return bleu.score


# ====================================================
# ====================== MAIN =========================
# ====================================================
def main():
    config = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ----- training -----
    train_cfg = config["training"]
    BATCH_SIZE = train_cfg["batch_size"]
    NUM_EPOCHS = train_cfg["num_epochs"]
    LEARNING_RATE = train_cfg["learning_rate"]

    BASE_TEACHER_FORCING = train_cfg["teacher_forcing"]
    MIN_TEACHER_FORCING = train_cfg.get("min_teacher_forcing", 0.1)
    TF_DECAY = train_cfg.get("tf_decay", 0.97)

    CLIP = train_cfg["clip"]
    MAX_LEN = train_cfg.get("max_len", 50)

    # early stopping (theo val_loss)
    EARLY_PATIENCE = train_cfg.get("early_patience", 3)
    EARLY_MIN_DELTA = train_cfg.get("early_min_delta", 1e-4)

    # ----- model -----
    model_cfg = config["model"]
    EMB_DIM = model_cfg["emb_dim"]
    HID_DIM = model_cfg["hid_dim"]
    N_LAYERS = model_cfg["n_layers"]
    DROPOUT = model_cfg["dropout"]

    # ----- data -----
    data_cfg = config["data"]
    data_dir = Path(data_cfg["data_dir"])

    # ----- vocab -----
    vocab_cfg = config["vocab"]
    MIN_FREQ = vocab_cfg["min_freq"]

    # ----- paths -----
    paths_cfg = config["paths"]
    ckpt_path = Path(paths_cfg["checkpoint"])
    ckpt_path.parent.mkdir(exist_ok=True, parents=True)

    results_dir = Path(paths_cfg.get("results", "../results/"))
    results_dir.mkdir(parents=True, exist_ok=True)

    # ============ LOAD DATA ===============
    print("Loading data...")
    train_src = read_lines(data_dir / "train.en")
    train_tgt = read_lines(data_dir / "train.fr")
    val_src = read_lines(data_dir / "val.en")
    val_tgt = read_lines(data_dir / "val.fr")

    train_src_tok = [tokenize_en(s) for s in train_src]
    train_tgt_tok = [tokenize_fr(s) for s in train_tgt]
    val_src_tok = [tokenize_en(s) for s in val_src]
    val_tgt_tok = [tokenize_fr(s) for s in val_tgt]

    src_vocab = build_vocab_from_token_lists(train_src_tok, min_freq=MIN_FREQ)
    tgt_vocab = build_vocab_from_token_lists(train_tgt_tok, min_freq=MIN_FREQ)

    print(f"SRC vocab size: {len(src_vocab)}")
    print(f"TGT vocab size: {len(tgt_vocab)}")

    train_ds = TranslationDataset(train_src_tok, train_tgt_tok, src_vocab, tgt_vocab)
    val_ds = TranslationDataset(val_src_tok, val_tgt_tok, src_vocab, tgt_vocab)

    collate = build_collate_fn(src_vocab, tgt_vocab)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate)

    # ============ INIT MODEL ===============
    pad_idx = tgt_vocab.stoi[PAD_TOKEN]
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

    enc = Encoder(len(src_vocab), EMB_DIM, HID_DIM, N_LAYERS, DROPOUT, pad_idx=src_vocab.stoi.get(PAD_TOKEN, 0))
    dec = AttentionDecoder(len(tgt_vocab), EMB_DIM, HID_DIM, N_LAYERS, DROPOUT)
    model = Seq2SeqWithAttention(enc, dec, device).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Scheduler theo val_loss
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=2,
    )

    # ============ LOGS ===============
    train_loss_log, val_loss_log, val_bleu_log = [], [], []

    best_bleu = -1.0                  # save checkpoint theo BLEU
    best_val_loss_report = float("inf")

    # Early stopping theo val_loss (dùng biến riêng)
    best_val_loss_es = float("inf")
    bad_epochs = 0

    # ============ TRAIN LOOP ===============
    print("Training model...")
    for epoch in range(1, NUM_EPOCHS + 1):
        t0 = time.time()

        teacher_forcing_ratio = max(
            MIN_TEACHER_FORCING,
            BASE_TEACHER_FORCING * (TF_DECAY ** (epoch - 1))
        )

        current_lr = optimizer.param_groups[0]["lr"]
        print(f"\n=== Epoch {epoch}/{NUM_EPOCHS} | TF={teacher_forcing_ratio:.3f} | LR={current_lr:.6f} ===")

        # ----- TRAIN -----
        train_loss = train_epoch(
            model, train_loader, optimizer, criterion,
            teacher_forcing_ratio=teacher_forcing_ratio,
            clip=CLIP,
            device=device,
        )

        # ----- VAL LOSS -----
        val_loss = evaluate_epoch(model, val_loader, criterion, device=device)

        # ----- VAL BLEU -----
        hyps, refs = generate_hyps_from_loader(
            model, val_loader, tgt_vocab,
            max_len=MAX_LEN,
            device=device,
        )
        val_bleu = compute_corpus_bleu(hyps, refs)

        # logs
        train_loss_log.append(train_loss)
        val_loss_log.append(val_loss)
        val_bleu_log.append(val_bleu)

        # scheduler step theo loss
        scheduler.step(val_loss)

        elapsed = time.time() - t0
        gap = abs(train_loss - val_loss)

        print(
            f"Epoch {epoch:02d}/{NUM_EPOCHS} | "
            f"Train Loss: {train_loss:.3f} | "
            f"Val Loss: {val_loss:.3f} | "
            f"Gap: {gap:.3f} | "
            f"Val BLEU: {val_bleu:.2f} | "
            f"Time: {elapsed:.1f}s"
        )

        # best val_loss (chỉ để report)
        if val_loss < best_val_loss_report:
            best_val_loss_report = val_loss

        # SAVE CHECKPOINT BY BLEU (đúng yêu cầu thầy + evaluate)
        if val_bleu > best_bleu:
            best_bleu = val_bleu
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "src_itos": src_vocab.itos,
                    "tgt_itos": tgt_vocab.itos,
                    "config": config,
                    "epoch": epoch,
                    "val_bleu": best_bleu,
                },
                ckpt_path,
            )
            print("  → Saved best checkpoint (by Val BLEU) to:", ckpt_path)

        # ======== EARLY STOPPING (by val_loss) ========
        if val_loss < best_val_loss_es - EARLY_MIN_DELTA:
            best_val_loss_es = val_loss
            bad_epochs = 0
        else:
            bad_epochs += 1

        if bad_epochs >= EARLY_PATIENCE:
            print(f"Early stopping: val_loss không giảm sau {EARLY_PATIENCE} epoch.")
            break

    # ============ SAVE METRICS (FOR PLOT) ===============
    metrics = {
        "train_loss": train_loss_log,
        "val_loss": val_loss_log,
        "val_bleu": val_bleu_log
    }
    metrics_path = results_dir / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("Saved metrics to:", metrics_path)

    print(
        f"\nTraining finished.\n"
        f"Best Val Loss (for report) = {best_val_loss_report:.3f}\n"
        f"Best Val BLEU (saved ckpt) = {best_bleu:.2f}"
    )


if __name__ == "__main__":
    main()
