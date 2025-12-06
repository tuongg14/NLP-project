import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import sacrebleu

from config_loader import load_config
from data import (
    read_lines, tokenize_en, tokenize_fr,
    build_vocab_from_token_lists, TranslationDataset, build_collate_fn,
    PAD_TOKEN, SOS_TOKEN, EOS_TOKEN,
)
from model import Seq2Seq, Encoder, Decoder

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
        output = model(src, src_lens, tgt_in,
                       teacher_forcing_ratio=teacher_forcing_ratio)
        vocab_size = output.size(-1)

        # Bỏ timestep 0 (ứng với <sos>)
        output = output[:, 1:, :].contiguous()    # [B, T-1, V]
        tgt_out = tgt_out[:, 1:].contiguous()     # [B, T-1]

        loss = criterion(
            output.view(-1, vocab_size),
            tgt_out.view(-1)
        )
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        epoch_loss += loss.item()

    return epoch_loss / len(dataloader)

# ====================================================
# =============== BLEU EVALUATION ====================
# ====================================================
def safe_tok_from_idx(vocab, idx: int):
    """
    Đổi id -> token, tránh lỗi out-of-range.
    vocab: instance Vocab trong data.py (có itos, stoi).
    """
    try:
        if 0 <= idx < len(vocab.itos):
            tok = vocab.itos[idx]
        else:
            tok = "<unk>"
    except Exception:
        tok = "<unk>"
    return tok

def generate_hyps_from_loader(model, dataloader, tgt_vocab,
                              max_len=50, device="cpu"):
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

            # ----- hypotheses (dự đoán) -----
            for seq in preds:
                tokens = []
                for idx in seq:
                    if idx == tgt_vocab.stoi[EOS_TOKEN]:
                        break
                    tok = safe_tok_from_idx(tgt_vocab, idx)
                    if tok not in (PAD_TOKEN, SOS_TOKEN, EOS_TOKEN):
                        tokens.append(tok)
                hyps.append(" ".join(tokens))

            # ----- references (ground truth) từ tgt_out -----
            tgt_np = tgt_out.cpu().numpy()
            for line in tgt_np:
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
    """
    Tính BLEU corpus-level bằng sacrebleu.
    hyps: list[str] – câu dự đoán
    refs: list[str] – câu tham chiếu
    """
    bleu = sacrebleu.corpus_bleu(hyps, [refs])
    return bleu.score

# ====================================================
# ====================== MAIN =========================
# ====================================================
def main():
    # ============ LOAD CONFIG ===============
    config = load_config()

    # device: lấy theo máy
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ----- training -----
    train_cfg = config["training"]
    BATCH_SIZE = train_cfg["batch_size"]
    NUM_EPOCHS = train_cfg["num_epochs"]
    LEARNING_RATE = train_cfg["learning_rate"]
    TEACHER_FORCING = train_cfg["teacher_forcing"]
    CLIP = train_cfg["clip"]

    # max_len cho greedy decode
    MAX_LEN = 50

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

    # ============ LOAD DATA ===============
    print("Loading data...")

    train_src = read_lines(data_dir / "train.en")
    train_tgt = read_lines(data_dir / "train.fr")
    val_src   = read_lines(data_dir / "val.en")
    val_tgt   = read_lines(data_dir / "val.fr")

    # Token hóa (giống notebook)
    train_src_tok = [tokenize_en(s) for s in train_src]
    train_tgt_tok = [tokenize_fr(s) for s in train_tgt]
    val_src_tok   = [tokenize_en(s) for s in val_src]
    val_tgt_tok   = [tokenize_fr(s) for s in val_tgt]

    # Build vocab (dùng min_freq từ config)
    src_vocab = build_vocab_from_token_lists(
        train_src_tok,
        min_freq=MIN_FREQ,
    )
    tgt_vocab = build_vocab_from_token_lists(
        train_tgt_tok,
        min_freq=MIN_FREQ,
    )

    print(f"SRC vocab size: {len(src_vocab)}")
    print(f"TGT vocab size: {len(tgt_vocab)}")

    # Dataset + Dataloader
    train_ds = TranslationDataset(train_src_tok, train_tgt_tok, src_vocab, tgt_vocab)
    val_ds   = TranslationDataset(val_src_tok, val_tgt_tok, src_vocab, tgt_vocab)

    collate = build_collate_fn(src_vocab, tgt_vocab)

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate,
    )

    # ============ INIT MODEL ===============
    pad_idx = tgt_vocab.stoi[PAD_TOKEN]
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

    enc = Encoder(len(src_vocab), EMB_DIM, HID_DIM, N_LAYERS, DROPOUT)
    dec = Decoder(len(tgt_vocab), EMB_DIM, HID_DIM, N_LAYERS, DROPOUT)
    model = Seq2Seq(enc, dec, device).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_bleu = -1.0

    # ============ TRAIN LOOP ===============
    print("Training model...")
    for epoch in range(1, NUM_EPOCHS + 1):
        t0 = time.time()

        train_loss = train_epoch(
            model, train_loader, optimizer, criterion,
            teacher_forcing_ratio=TEACHER_FORCING,
            clip=CLIP,
            device=device,
        )

        hyps, refs = generate_hyps_from_loader(
            model, val_loader, tgt_vocab,
            max_len=MAX_LEN,
            device=device,
        )
        val_bleu = compute_corpus_bleu(hyps, refs)

        elapsed = time.time() - t0

        print(
            f"Epoch {epoch:02d}/{NUM_EPOCHS} | "
            f"Train Loss: {train_loss:.3f} | "
            f"Val BLEU: {val_bleu:.2f} | "
            f"Time: {elapsed:.1f}s"
        )

        # Lưu checkpoint tốt nhất theo BLEU
        if val_bleu > best_bleu:
            best_bleu = val_bleu
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "src_itos": src_vocab.itos,
                    "tgt_itos": tgt_vocab.itos,
                    "config": config,
                },
                ckpt_path,
            )
            print("  → Saved best checkpoint to:", ckpt_path)

    print(f"\nTraining finished. BEST VAL BLEU = {best_bleu:.2f}")

if __name__ == "__main__":
    main()
