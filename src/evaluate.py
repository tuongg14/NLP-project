# evaluate.py

import torch
from pathlib import Path

from tqdm import tqdm

from config_loader import load_config
from data import (
    read_lines, tokenize_en, tokenize_fr,
    TranslationDataset, build_collate_fn,
    Vocab, PAD_TOKEN, SOS_TOKEN, EOS_TOKEN,
)
from model import Encoder, Decoder, Seq2Seq
from train import generate_hyps_from_loader, compute_corpus_bleu


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def vocab_from_itos(itos_list):
    """
    Tạo lại Vocab từ list itos đã save trong checkpoint.
    """
    v = Vocab(tokens_list=[])
    v.itos = itos_list
    v.stoi = {tok: i for i, tok in enumerate(itos_list)}
    return v


def run_evaluate(
    ckpt_path="../checkpoints/best_model.pth",
    save_samples_path="../results/samples.txt",
    max_len=50,
):
    config = load_config()

    data_cfg = config["data"]
    data_dir = Path(data_cfg["data_dir"])

    # ----- load test data -----
    test_src = read_lines(data_dir / "test_2018_flickr.en")
    test_tgt = read_lines(data_dir / "test_2018_flickr.fr")

    test_src_tok = [tokenize_en(s) for s in test_src]
    test_tgt_tok = [tokenize_fr(s) for s in test_tgt]

    print("Loading checkpoint:", ckpt_path)
    ckpt = torch.load(ckpt_path, map_location=DEVICE)

    # ----- rebuild vocab from checkpoint -----
    src_vocab = vocab_from_itos(ckpt["src_itos"])
    tgt_vocab = vocab_from_itos(ckpt["tgt_itos"])

    # ----- rebuild model from config in checkpoint -----
    cfg_ckpt = ckpt.get("config", config)
    m_cfg = cfg_ckpt["model"]
    emb_dim = m_cfg["emb_dim"]
    hid_dim = m_cfg["hid_dim"]
    n_layers = m_cfg["n_layers"]
    dropout = m_cfg["dropout"]

    enc = Encoder(len(src_vocab), emb_dim, hid_dim, n_layers, dropout)
    dec = Decoder(len(tgt_vocab), emb_dim, hid_dim, n_layers, dropout)
    model = Seq2Seq(enc, dec, DEVICE).to(DEVICE)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # ----- build test loader -----
    test_ds = TranslationDataset(test_src_tok, test_tgt_tok, src_vocab, tgt_vocab)
    collate_fn = build_collate_fn(src_vocab, tgt_vocab)

    from torch.utils.data import DataLoader
    test_loader = DataLoader(
        test_ds,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        collate_fn=collate_fn,
    )

    # ----- generate hyps & refs -----
    print("Generating hypotheses on test set...")
    hyps_test, refs_test = generate_hyps_from_loader(
        model, test_loader, tgt_vocab, max_len=max_len, device=DEVICE
    )

    print("Computing BLEU on test set...")
    test_bleu = compute_corpus_bleu(hyps_test, refs_test)
    print("Test BLEU:", test_bleu)

    # ----- save samples -----
    results_dir = Path(save_samples_path).parent
    results_dir.mkdir(exist_ok=True, parents=True)

    print("Saving sample translations to:", save_samples_path)
    with open(save_samples_path, "w", encoding="utf-8") as f:
        for src_tokens, ref, hyp in zip(test_src_tok[:200], refs_test[:200], hyps_test[:200]):
            f.write("SRC: " + " ".join(src_tokens) + "\n")
            f.write("REF: " + ref + "\n")
            f.write("HYP: " + hyp + "\n\n")

    return test_bleu


if __name__ == "__main__":
    run_evaluate()
