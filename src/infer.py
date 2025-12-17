# infer.py
import torch
from pathlib import Path

from config_loader import load_config
from data import tokenize_en, SOS_TOKEN, EOS_TOKEN, PAD_TOKEN
from model import Encoder, Decoder, Seq2Seq, AttentionDecoder, Seq2SeqWithAttention

# =====================
# Device & paths
# =====================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# =====================
# Helpers
# =====================
def build_stoi(itos):
    """
    Build string-to-index dictionary từ itos (index-to-string).
    """
    return {tok: i for i, tok in enumerate(itos)}

def numericalize(tokens, stoi, unk_token="<unk>"):
    """
    Chuyển list token -> list index.
    Token không có trong vocab sẽ map về <unk>.
    """
    unk_idx = stoi.get(unk_token, stoi.get("<unk>", 3))
    return [stoi.get(t, unk_idx) for t in tokens]

def is_attention_checkpoint(state_dict, emb_dim):
    """
    AttentionDecoder:
        input_size = emb_dim + hid_dim
    Non-attention:
        input_size = emb_dim
    """
    for k, v in state_dict.items():
        if "decoder.rnn.weight_ih_l0" in k:
            return v.shape[1] > emb_dim
    return False

# =====================
# Load checkpoint
# =====================
config = load_config()
CKPT_PATH = PROJECT_ROOT / config["paths"]["checkpoint"]

print("Loading checkpoint:", CKPT_PATH)
ckpt = torch.load(CKPT_PATH, map_location=DEVICE)

# ---- vocab ----
src_itos = ckpt["src_itos"]
tgt_itos = ckpt["tgt_itos"]

src_stoi = build_stoi(src_itos)
tgt_stoi = build_stoi(tgt_itos)

# pad index cho encoder embedding
src_pad_idx = src_stoi.get(PAD_TOKEN, 0)

# ---- config from checkpoint (IMPORTANT) ----
cfg = ckpt["config"]
m_cfg = cfg["model"]

emb_dim = m_cfg["emb_dim"]
hid_dim = m_cfg["hid_dim"]
n_layers = m_cfg["n_layers"]
dropout = m_cfg["dropout"]

# =====================
# Build model (same as evaluate)
# =====================
use_attn = is_attention_checkpoint(
    ckpt["model_state_dict"],
    emb_dim
)

enc = Encoder(
    len(src_itos),
    emb_dim,
    hid_dim,
    n_layers,
    dropout,
    pad_idx=src_pad_idx
)

if use_attn:
    dec = AttentionDecoder(
        len(tgt_itos),
        emb_dim,
        hid_dim,
        n_layers,
        dropout
    )
    model = Seq2SeqWithAttention(enc, dec, DEVICE).to(DEVICE)
    print("→ Detected model type: Seq2SeqWithAttention (Luong)")
else:
    dec = Decoder(
        len(tgt_itos),
        emb_dim,
        hid_dim,
        n_layers,
        dropout
    )
    model = Seq2Seq(enc, dec, DEVICE).to(DEVICE)
    print("→ Detected model type: Seq2Seq (no attention)")

# ---- load weights ----
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

print("✓ Model loaded successfully")

# =====================
# REQUIRED FUNCTION (rubric)
# =====================
def translate(sentence: str, max_len: int = 50) -> str:
    """
    Translate a single English sentence to target language.
    """
    # Tokenize English sentence
    tokens = tokenize_en(sentence)

    # Convert tokens to indices
    ids = numericalize(tokens, src_stoi, unk_token="<unk>")

    # Build tensor: [1, S]
    src_tensor = torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(DEVICE)
    src_len = torch.tensor([len(ids)], dtype=torch.long).to(DEVICE)

    sos_idx = tgt_stoi[SOS_TOKEN]
    eos_idx = tgt_stoi[EOS_TOKEN]

    # Greedy decoding (no teacher forcing)
    with torch.no_grad():
        preds = model.greedy_decode(
            src_tensor,
            src_len,
            max_len=max_len,
            sos_idx=sos_idx,
            eos_idx=eos_idx
        )

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

# =====================
# CLI demo
# =====================
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
