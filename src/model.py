import torch
import torch.nn as nn
import random


class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers=1, dropout=0., pad_idx=0):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim, padding_idx=pad_idx)

        self.rnn = nn.LSTM(
            emb_dim,
            hid_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_lengths):
        # src: [B, S]
        embedded = self.dropout(self.embedding(src))  # [B, S, E]

        # pack padded
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded,
            src_lengths.cpu(),
            batch_first=True,
            enforce_sorted=False
        )

        _, (hidden, cell) = self.rnn(packed)
        # hidden, cell: [n_layers, B, hid_dim]
        return hidden, cell


class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers=1, dropout=0., pad_idx=0):
        super().__init__()

        self.embedding = nn.Embedding(output_dim, emb_dim, padding_idx=pad_idx)

        self.rnn = nn.LSTM(
            emb_dim,
            hid_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout
        )

        self.fc_out = nn.Linear(hid_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_step, hidden, cell):
        # input_step: [B, 1]
        embedded = self.dropout(self.embedding(input_step))  # [B, 1, emb]

        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))  # output: [B, 1, hid]

        pred = self.fc_out(output.squeeze(1))  # [B, output_dim]

        return pred, hidden, cell


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, src_len, tgt_in, teacher_forcing_ratio=0.5):
        batch_size = src.size(0)
        tgt_len = tgt_in.size(1)
        vocab_size = self.decoder.fc_out.out_features

        outputs = torch.zeros(batch_size, tgt_len, vocab_size).to(self.device)

        hidden, cell = self.encoder(src, src_len)

        input_tok = tgt_in[:, 0].unsqueeze(1)  # SOS

        for t in range(1, tgt_len):
            pred, hidden, cell = self.decoder(input_tok, hidden, cell)
            outputs[:, t] = pred

            teacher_force = random.random() < teacher_forcing_ratio
            top1 = pred.argmax(1).unsqueeze(1)

            input_tok = tgt_in[:, t].unsqueeze(1) if teacher_force else top1

        return outputs

    def greedy_decode(self, src, src_len, max_len=50, sos_idx=None, eos_idx=None):
        batch_size = src.size(0)

        hidden, cell = self.encoder(src, src_len)

        assert sos_idx is not None and eos_idx is not None

        input_tok = torch.LongTensor([[sos_idx]] * batch_size).to(self.device)
        preds = [[] for _ in range(batch_size)]
        finished = [False] * batch_size

        for _ in range(max_len):
            pred, hidden, cell = self.decoder(input_tok, hidden, cell)
            top1 = pred.argmax(1)  # [B]
            input_tok = top1.unsqueeze(1)

            for i in range(batch_size):
                if not finished[i]:
                    token_id = top1[i].item()
                    if token_id == eos_idx:
                        finished[i] = True
                    preds[i].append(token_id)

        return preds
