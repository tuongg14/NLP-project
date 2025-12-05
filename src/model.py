import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers=1, dropout=0.1, bidirectional=False):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.lstm = nn.LSTM(emb_dim, hid_dim, num_layers=n_layers, batch_first=True, bidirectional=bidirectional)
        self.dropout = nn.Dropout(dropout)
        self.bidirectional = bidirectional


    def forward(self, src, src_lengths):
        embedded = self.dropout(self.embedding(src))
        packed = pack_padded_sequence(embedded, src_lengths.cpu(), batch_first=True, enforce_sorted=True)
        packed_out, (h, c) = self.lstm(packed)
        out, out_lengths = pad_packed_sequence(packed_out, batch_first=True)
        return out, (h, c)
    
class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers=1, dropout=0.1, padding_idx=0):
        super().__init__()
        self.output_dim = output_dim
        self.embedding = nn.Embedding(output_dim, emb_dim, padding_idx=padding_idx)
        self.lstm = nn.LSTM(emb_dim, hid_dim, num_layers=n_layers, batch_first=True)
        self.fc_out = nn.Linear(hid_dim, output_dim)
        self.dropout = nn.Dropout(dropout)


    def forward_step(self, input_tokens, hidden):
        # input_tokens: (batch, )
        input_tokens = input_tokens.unsqueeze(1)
        emb = self.dropout(self.embedding(input_tokens))
        out, hidden = self.lstm(emb, hidden)
        out = out.squeeze(1)
        logits = self.fc_out(out)
        return logits, hidden
    
class Seq2Seq(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder, device: str):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, src_lengths, tgt, teacher_forcing_ratio=0.5):
        batch_size = src.size(0)
        max_tgt_len = tgt.size(1)
        vocab_size = self.decoder.output_dim


        outputs = torch.zeros(batch_size, max_tgt_len, vocab_size).to(self.device)


        enc_out, hidden = self.encoder(src, src_lengths)
        dec_hidden = hidden
        input_tok = tgt[:, 0]


        for t in range(1, max_tgt_len):
            logits, dec_hidden = self.decoder.forward_step(input_tok, dec_hidden)
            outputs[:, t, :] = logits
            use_teacher = torch.rand(1).item() < teacher_forcing_ratio
            top1 = logits.argmax(1)
            input_tok = tgt[:, t] if use_teacher else top1


        return outputs

    def greedy_decode(self, src, src_lengths, sos_idx, eos_idx, max_len=50):
        enc_out, hidden = self.encoder(src, src_lengths)
        dec_hidden = hidden
        batch_size = src.size(0)
        input_tok = torch.tensor([sos_idx] * batch_size, dtype=torch.long, device=self.device)


        results = [[] for _ in range(batch_size)]
        finished = [False] * batch_size


        for _ in range(max_len):
            logits, dec_hidden = self.decoder.forward_step(input_tok, dec_hidden)
            top1 = logits.argmax(1)
            input_tok = top1
            for i in range(batch_size):
                if not finished[i]:
                    tok = top1[i].item()
                    results[i].append(tok)
                    if tok == eos_idx:
                        finished[i] = True
            if all(finished):
                break
        return results