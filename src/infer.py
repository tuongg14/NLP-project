# infer.py
import torch
from pathlib import Path

from config_loader import load_config
from data import tokenize_en, SOS_TOKEN, EOS_TOKEN, PAD_TOKEN
from model import Encoder, Decoder, Seq2Seq, AttentionDecoder, Seq2SeqWithAttention

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===== helpers =====
def build_stoi(itos):
    return {tok: i for i, tok in enumerate(itos)}

def numericalize(tokens, stoi, unk_token="<unk>"):
    unk_idx = stoi.get(unk_token, stoi.get("<unk>", 3))
    return [stoi.get(t, unk_idx) for t in tokens]

def _state_dict_looks_like_attention(state_dict: dict) -> bool:
    for k in state_dict.keys():
        if "attention" in k.lower() or "attn" in k.lower():
            return True
    return False

def build_model_from_config(config: dict, src_vocab_size: int, tgt_vocab_size: int, use_attention: bool):
    mcfg = config["model"]
    emb_dim = mcfg["emb_dim"]
    hid_dim = mcfg["hid_dim"]
    n_layers = mcfg["n_layers"]
    dropout = mcfg["dropout"]

    enc = Encoder(src_vocab_size, emb_dim, hid_dim, n_layers=n_layers, dropout=dropout)

    if use_attention:
        dec = AttentionDecoder(tgt_vocab_size, emb_dim, hid_dim, n_layers=n_layers, dropout=dropout)
        model = Seq2SeqWithAttention(enc, dec, DEVICE).to(DEVICE)
    else:
        dec = Decoder(tgt_vocab_size, emb_dim, hid_dim, n_layers=n_layers, dropout=dropout)
        model = Seq2Seq(enc, dec, DEVICE).to(DEVICE)

    return model

# ===== load once =====
CKPT_PATH = "../checkpoints/best_model.pth"

print("Loading checkpoint:", CKPT_PATH)
ckpt = torch.load(CKPT_PATH, map_location=DEVICE)

src_itos = ckpt["src_itos"]
tgt_itos = ckpt["tgt_itos"]
src_stoi = build_stoi(src_itos)
tgt_stoi = build_stoi(tgt_itos)

config = ckpt.get("config", None)
if config is None:
    config = load_config()

use_attention = _state_dict_looks_like_attention(ckpt["model_state_dict"])
model = build_model_from_config(config, len(src_itos), len(tgt_itos), use_attention)
model.load_state_dict(ckpt["model_state_dict"])
model.to(DEVICE)
model.eval()

print(f"Loaded model. Attention = {use_attention}")

# ===== REQUIRED FUNCTION (rubric) =====
def translate(sentence: str, max_len: int = 50) -> str:
    """
    REQUIRED: translate(sentence: str) -> str  
    Tokenize -> tensor -> encoder -> greedy decode -> detokenize
    Stop when <eos> or max_len
    """
    tokens = tokenize_en(sentence)
    ids = numericalize(tokens, src_stoi, unk_token="<unk>")

    src_tensor = torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(DEVICE)  # [1,S]
    src_len = torch.tensor([len(ids)], dtype=torch.long).to(DEVICE)           # [1]

    sos_idx = tgt_stoi[SOS_TOKEN]
    eos_idx = tgt_stoi[EOS_TOKEN]

    with torch.no_grad():
        preds = model.greedy_decode(
            src_tensor, src_len,
            max_len=max_len,
            sos_idx=sos_idx,
            eos_idx=eos_idx
        )

    # preds: Tensor [1,T] hoặc list[list[int]]
    pred_ids = preds[0].tolist() if torch.is_tensor(preds) else preds[0]

    words = []
    for idx in pred_ids:
        if idx == eos_idx:
            break
        tok = tgt_itos[idx] if 0 <= idx < len(tgt_itos) else "<unk>"
        if tok in (SOS_TOKEN, PAD_TOKEN, EOS_TOKEN):
            continue
        words.append(tok)

    return " ".join(words)

# ===== CLI demo =====
def demo():
    print("\n=== INFERENCE DEMO ===")
    print("Type 'q' to quit.")
    while True:
        s = input("\nEN> ").strip()
        if s.lower() == "q":
            break
        print("FR>", translate(s))

if __name__ == "__main__":
    demo()
