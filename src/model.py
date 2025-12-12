import torch
import torch.nn as nn
import random
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence

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

        packed_outputs, (hidden, cell) = self.rnn(packed)
        # unpack outputs to get encoder outputs (padding restored)
        encoder_outputs, _ = pad_packed_sequence(packed_outputs, batch_first=True)

        # hidden, cell: [n_layers, B, hid_dim]
        return encoder_outputs, (hidden, cell)


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

        #hidden, cell = self.encoder(src, src_len)

        enc_out = self.encoder(src, src_len)
        # encoder may return either (hidden, cell) or (encoder_outputs, (hidden, cell))
        if isinstance(enc_out, tuple) and len(enc_out) == 2 and isinstance(enc_out[0], torch.Tensor):
            # new encoder: returned (encoder_outputs, (hidden, cell))
            _, (hidden, cell) = enc_out
        else:
            # old encoder: returned (hidden, cell)
            hidden, cell = enc_out

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

class LuongAttention(nn.Module):
    def __init__(self, hid_dim):
        super().__init__()
        self.hid_dim = hid_dim

    def forward(self, decoder_hidden, encoder_outputs, mask=None):
        """
        decoder_hidden: [B, H]
        encoder_outputs: [B, S, H]
        mask: [B, S] (optional)
        """

        # energy = dot(dec_hidden, enc_out)
        energy = torch.bmm(
            encoder_outputs,
            decoder_hidden.unsqueeze(2)
        ).squeeze(2)   # [B, S]

        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)

        attn_weights = F.softmax(energy, dim=1)   # [B, S]

        context = torch.bmm(
            attn_weights.unsqueeze(1),
            encoder_outputs
        ).squeeze(1)   # [B, H]

        return context, attn_weights
    
class AttentionDecoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers=1, dropout=0.3):
        super().__init__()

        self.output_dim = output_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers

        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.attention = LuongAttention(hid_dim)

        self.rnn = nn.LSTM(
            emb_dim + hid_dim,
            hid_dim,
            num_layers=n_layers,
            dropout=dropout if n_layers > 1 else 0,
            batch_first=True
        )

        self.fc_out = nn.Linear(hid_dim * 2, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, cell, encoder_outputs, mask=None):
        """
        input: [B]
        hidden, cell: [n_layers, B, H]
        encoder_outputs: [B, S, H]
        """

        input = input.unsqueeze(1)   # [B, 1]
        emb = self.dropout(self.embedding(input))  # [B, 1, E]

        # attention dùng hidden layer cuối
        context, attn_weights = self.attention(
            hidden[-1], encoder_outputs, mask
        )  # context: [B, H]

        context = context.unsqueeze(1)  # [B, 1, H]

        rnn_input = torch.cat((emb, context), dim=2)

        output, (hidden, cell) = self.rnn(
            rnn_input, (hidden, cell)
        )

        output = output.squeeze(1)   # [B, H]
        context = context.squeeze(1)

        pred = self.fc_out(
            torch.cat((output, context), dim=1)
        )  # [B, vocab]

        return pred, hidden, cell, attn_weights
    
class Seq2SeqWithAttention(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, src_lens, tgt,
                teacher_forcing_ratio=0.5):

        B, T = tgt.size()
        vocab_size = self.decoder.output_dim

        outputs = torch.zeros(B, T, vocab_size).to(self.device)

        encoder_outputs, (hidden, cell) = self.encoder(src, src_lens)

        input = tgt[:, 0]  # <sos>

        for t in range(1, T):
            pred, hidden, cell, _ = self.decoder(
                input, hidden, cell, encoder_outputs
            )

            outputs[:, t] = pred

            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            top1 = pred.argmax(1)

            input = tgt[:, t] if teacher_force else top1

        return outputs

    def greedy_decode(self, src, src_lens,
                      max_len, sos_idx, eos_idx):

        encoder_outputs, (hidden, cell) = self.encoder(src, src_lens)

        input = torch.full(
            (src.size(0),),
            sos_idx,
            dtype=torch.long,
            device=self.device
        )

        outputs = []

        for _ in range(max_len):
            pred, hidden, cell, _ = self.decoder(
                input, hidden, cell, encoder_outputs
            )

            top1 = pred.argmax(1)
            outputs.append(top1)
            input = top1

        outputs = torch.stack(outputs, dim=1)
        return outputs