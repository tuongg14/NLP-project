# evaluate.py
import torch
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm
import sacrebleu

from config_loader import load_config
from data import (
    read_lines, tokenize_en, tokenize_fr,
    TranslationDataset, build_collate_fn,
    Vocab, PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN
)
from model import Encoder, Decoder, Seq2Seq, AttentionDecoder, Seq2SeqWithAttention


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------
# Helpers
# ---------------------------
def vocab_from_itos(itos_list):
    """
    Tạo lại Vocab từ list itos đã save trong checkpoint
    mà KHÔNG gọi __init__ (tránh rebuild).
    """
    v = Vocab.__new__(Vocab)          # bypass __init__
    v.itos = list(itos_list)
    v.stoi = {tok: i for i, tok in enumerate(v.itos)}
    return v


def safe_tok_from_idx(vocab, idx: int):
    if 0 <= idx < len(vocab.itos):
        return vocab.itos[idx]
    return UNK_TOKEN


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

            # references từ tgt_out
            tgt_np = tgt_out.cpu().numpy()
            for line in tgt_np:
                tokens = []
                for idx in line:
                    if int(idx) == tgt_vocab.stoi[EOS_TOKEN]:
                        break
                    tok = safe_tok_from_idx(tgt_vocab, int(idx))
                    if tok not in (PAD_TOKEN, SOS_TOKEN, EOS_TOKEN):
                        tokens.append(tok)
                refs.append(" ".join(tokens))

    return hyps, refs


def compute_corpus_bleu(hyps, refs):
    bleu = sacrebleu.corpus_bleu(hyps, [refs], force=True)
    return bleu.score


def pick_existing_file(data_dir: Path, candidates):
    """
    Chọn file tồn tại đầu tiên trong list candidates.
    """
    for name in candidates:
        p = data_dir / name
        if p.exists():
            return p
    return None


def load_checkpoint_safely(ckpt_path: Path):
    """
    Giảm warning torch.load nếu torch version hỗ trợ weights_only.
    """
    try:
        return torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
    except TypeError:
        return torch.load(ckpt_path, map_location=DEVICE)


def is_attention_checkpoint(state_dict):
    """
    Heuristic: checkpoint có attention nếu có key chứa 'attention'
    """
    for k in state_dict.keys():
        if "attention" in k.lower():
            return True
    return False


# ---------------------------
# Main evaluate
# ---------------------------
def run_evaluate(
    ckpt_path="../checkpoints/best_model.pth",
    save_samples_path="../results/samples.txt",
    max_len=50,
):
    config = load_config()

    data_dir = Path(config["data"]["data_dir"])
    ckpt_path = Path(ckpt_path)

    # ---- pick test files (auto-detect) ----
    test_en = pick_existing_file(data_dir, ["test.en", "test_2018_flickr.en", "test_2017_flickr.en"])
    test_fr = pick_existing_file(data_dir, ["test.fr", "test_2018_flickr.fr", "test_2017_flickr.fr"])

    if test_en is None or test_fr is None:
        raise FileNotFoundError(
            f"Không tìm thấy file test trong {data_dir}.\n"
            f"Cần có 1 cặp như: test.en/test.fr hoặc test_2018_flickr.en/.fr"
        )

    # ---- load test data ----
    test_src = read_lines(test_en)
    test_tgt = read_lines(test_fr)

    test_src_tok = [tokenize_en(s) for s in test_src]
    test_tgt_tok = [tokenize_fr(s) for s in test_tgt]

    # ---- load checkpoint ----
    print("Loading checkpoint:", ckpt_path)
    ckpt = load_checkpoint_safely(ckpt_path)

    # ---- rebuild vocab from checkpoint ----
    src_vocab = vocab_from_itos(ckpt["src_itos"])
    tgt_vocab = vocab_from_itos(ckpt["tgt_itos"])

    # ---- rebuild model from ckpt config ----
    cfg = ckpt.get("config", config)
    m_cfg = cfg["model"]
    emb_dim = m_cfg["emb_dim"]
    hid_dim = m_cfg["hid_dim"]
    n_layers = m_cfg["n_layers"]
    dropout = m_cfg["dropout"]

    # Decide attention vs non-attention based on state_dict
    use_attn = is_attention_checkpoint(ckpt["model_state_dict"])

    enc = Encoder(len(src_vocab), emb_dim, hid_dim, n_layers, dropout)

    if use_attn:
        dec = AttentionDecoder(len(tgt_vocab), emb_dim, hid_dim, n_layers, dropout)
        model = Seq2SeqWithAttention(enc, dec, DEVICE).to(DEVICE)
        print("→ Detected model type: Seq2SeqWithAttention (Luong)")
    else:
        dec = Decoder(len(tgt_vocab), emb_dim, hid_dim, n_layers, dropout)
        model = Seq2Seq(enc, dec, DEVICE).to(DEVICE)
        print("→ Detected model type: Seq2Seq (no attention)")

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # ---- build test loader ----
    test_ds = TranslationDataset(test_src_tok, test_tgt_tok, src_vocab, tgt_vocab)
    collate_fn = build_collate_fn(src_vocab, tgt_vocab)

    test_loader = DataLoader(
        test_ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
        collate_fn=collate_fn,
    )

    # ---- generate & BLEU ----
    print("Generating hypotheses on test set...")
    hyps_test, refs_test = generate_hyps_from_loader(
        model, test_loader, tgt_vocab, max_len=max_len, device=DEVICE
    )

    print("Computing BLEU on test set...")
    test_bleu = compute_corpus_bleu(hyps_test, refs_test)
    print("Test BLEU:", test_bleu)

    # ---- save samples ----
    save_samples_path = Path(save_samples_path)
    save_samples_path.parent.mkdir(parents=True, exist_ok=True)

    print("Saving sample translations to:", save_samples_path)
    with open(save_samples_path, "w", encoding="utf-8") as f:
        for src_tokens, ref, hyp in zip(test_src_tok[:200], refs_test[:200], hyps_test[:200]):
            f.write("SRC: " + " ".join(src_tokens) + "\n")
            f.write("REF: " + ref + "\n")
            f.write("HYP: " + hyp + "\n\n")

    return test_bleu


if __name__ == "__main__":
    run_evaluate()
