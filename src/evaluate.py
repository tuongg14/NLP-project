# evaluate.py

import torch
from pathlib import Path
from tqdm import tqdm

from model import Seq2Seq
from data import src_vocab, tgt_vocab, SOS_TOKEN, EOS_TOKEN, PAD_TOKEN
from data import test_loader  # test_loader đã được khởi tạo trong data.py
from model import model       # model đã khởi tạo trong model.py
from train import generate_hyps_from_loader, compute_corpus_bleu

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_evaluate(ckpt_path="../checkpoints/best_model.pt",
                 save_samples_path="../results/samples.txt",
                 max_len=50):
    """
    Chạy evaluate BLEU + lưu mẫu dịch.
    Giống hệt Cell 12 trong notebook của bạn.
    """

    print("Loading checkpoint:", ckpt_path)
    ckpt = torch.load(ckpt_path, map_location=DEVICE)

    # load state_dict
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(DEVICE)

    print("Generating hypotheses...")
    hyps_test, refs_test = generate_hyps_from_loader(
        model,
        test_loader,
        tgt_vocab,
        max_len=max_len
    )

    print("Computing BLEU...")
    test_bleu = compute_corpus_bleu(hyps_test, refs_test)
    print("Test BLEU:", test_bleu)

    print("Saving sample translations...")

    results_dir = Path(save_samples_path).parent
    results_dir.mkdir(exist_ok=True)

    with open(save_samples_path, "w", encoding="utf-8") as f:
        for src_tok, ref, hyp in zip(test_loader.dataset.src_tok[:200],
                                     refs_test[:200],
                                     hyps_test[:200]):
            f.write("SRC: " + " ".join(src_tok) + "\n")
            f.write("REF: " + ref + "\n")
            f.write("HYP: " + hyp + "\n\n")

    print("Samples saved to:", save_samples_path)
    return test_bleu
