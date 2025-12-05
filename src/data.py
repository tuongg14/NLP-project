from pathlib import Path
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

class TranslationDataset(Dataset):
    def __init__(self, src_lines, tgt_lines):
        assert len(src_lines) == len(tgt_lines)
        self.src = src_lines
        self.tgt = tgt_lines


    def __len__(self):
        return len(self.src)


    def __getitem__(self, idx):
        return self.src[idx], self.tgt[idx]

def read_lines(path: Path, max_examples=None):
    with open(path, 'r', encoding='utf-8') as f:
        lines = [l.strip() for l in f if l.strip()]
    if max_examples:
        return lines[:max_examples]
    return lines

def build_tokenizers():
    en_tokenizer = get_tokenizer('spacy', language='en_core_web_sm')
    fr_tokenizer = get_tokenizer('spacy', language='fr_core_news_sm')
    return en_tokenizer, fr_tokenizer

def build_vocabs(en_tokenizer, fr_tokenizer, src_lines, tgt_lines, config):
    specials = config['vocab'].get('specials', ['<unk>', '<pad>', '<sos>', '<eos>'])
    max_size = config['vocab'].get('max_size', 10000)

    def yield_en():
        for s in src_lines:
            yield en_tokenizer(s)


    def yield_fr():
        for t in tgt_lines:
            yield fr_tokenizer(t)


    vocab_en = build_vocab_from_iterator(yield_en(), max_tokens=max_size, specials=specials)
    vocab_fr = build_vocab_from_iterator(yield_fr(), max_tokens=max_size, specials=specials)
    vocab_en.set_default_index(vocab_en['<unk>'])
    vocab_fr.set_default_index(vocab_fr['<unk>'])
    return vocab_en, vocab_fr

def numericalize(text, tokenizer, vocab, add_sos_eos=False, sos='<sos>', eos='<eos>'):
    toks = tokenizer(text)
    ids = [vocab[t] for t in toks]
    if add_sos_eos:
        ids = [vocab[sos]] + ids + [vocab[eos]]
    return torch.tensor(ids, dtype=torch.long)

def collate_fn(batch, en_tokenizer, fr_tokenizer, vocab_en, vocab_fr, device):
    src_tensors = [numericalize(x[0], en_tokenizer, vocab_en, add_sos_eos=False) for x in batch]
    tgt_tensors = [numericalize(x[1], fr_tokenizer, vocab_fr, add_sos_eos=True) for x in batch]

    src_lengths = torch.tensor([t.size(0) for t in src_tensors], dtype=torch.long)
    src_lengths, perm = src_lengths.sort(descending=True)
    src_tensors = [src_tensors[i] for i in perm]
    tgt_tensors = [tgt_tensors[i] for i in perm]

    src_padded = pad_sequence(src_tensors, batch_first=True, padding_value=vocab_en['<pad>']).to(device)
    tgt_padded = pad_sequence(tgt_tensors, batch_first=True, padding_value=vocab_fr['<pad>']).to(device)
    tgt_lengths = torch.tensor([t.size(0) for t in tgt_tensors], dtype=torch.long).to(device)

    return src_padded, src_lengths.to(device), tgt_padded, tgt_lengths

def load_datasets_and_vocabs(config, device):
    data_dir = Path(config['data']['data_dir'])
    max_examples = config['data'].get('max_examples', None)
    train_src = read_lines(data_dir / config['data']['train_src'], max_examples)
    train_tgt = read_lines(data_dir / config['data']['train_tgt'], max_examples)
    val_src = read_lines(data_dir / config['data']['val_src'], max_examples)
    val_tgt = read_lines(data_dir / config['data']['val_tgt'], max_examples)

    en_tok, fr_tok = build_tokenizers()
    vocab_en, vocab_fr = build_vocabs(en_tok, fr_tok, train_src, train_tgt, config)


    train_ds = TranslationDataset(train_src, train_tgt)
    val_ds = TranslationDataset(val_src, val_tgt)
    return train_ds, val_ds, en_tok, fr_tok, vocab_en, vocab_fr