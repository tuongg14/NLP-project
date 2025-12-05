# infer.py

import torch
from pathlib import Path

from model import model
from data import (
    src_vocab, tgt_vocab,
    SOS_TOKEN, EOS_TOKEN, PAD_TOKEN,
    tokenize_en,
)
from model import Seq2Seq  # để type check

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_checkpoint(ckpt_path="../checkpoints/best_model.pt"):
    """
    Load checkpoint giống hệt Cell 12 notebook.
    """
    print("Loading checkpoint:", ckpt_path)
    ckpt = torch.load(ckpt_path, map_location=DEVICE)

    model.load_state_dict(ckpt["model_state_dict"])
    model.to(DEVICE)
    model.eval()
    return model


def translate_sentence(sentence, model, src_vocab, tgt_vocab, max_len=50):
    """
    Y hệt Cell 14 trong notebook của bạn.
    """

    model.eval()
    tokens = tokenize_en(sentence)
    ids = src_vocab.numericalize(tokens)
    src_tensor = torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(DEVICE)
    src_len = torch.tensor([len(ids)], dtype=torch.long).to(DEVICE)

    with torch.no_grad():
        preds = model.greedy_decode(
            src_tensor,
            src_len,
            max_len=max_len,
            sos_idx=tgt_vocab.stoi[SOS_TOKEN],
            eos_idx=tgt_vocab.stoi[EOS_TOKEN]
        )[0]

    words = []
    for idx in preds:
        if idx == tgt_vocab.stoi[EOS_TOKEN]:
            break
        if idx < len(tgt_vocab.itos):
            words.append(tgt_vocab.itos[idx])
        else:
            words.append("<unk>")

    return " ".join(words)


def translate_batch(sentences, model, max_len=50):
    """
    Dịch nhiều câu (list[str]) → list[str].
    """
    results = []
    for sent in sentences:
        hyp = translate_sentence(sent, model, src_vocab, tgt_vocab, max_len)
        results.append(hyp)
    return results


def demo():
    """
    Demo CLI: chạy thử dịch 1 câu.
    """
    load_checkpoint()
    while True:
        txt = input("\nEnter sentence (or 'q' to quit): ")
        if txt.lower().strip() == "q":
            break
        print("→", translate_sentence(txt, model, src_vocab, tgt_vocab))


if __name__ == "__main__":
    demo()
