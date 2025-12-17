# model.py
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


# =========================
# Encoder
# =========================
class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers=1, dropout=0.0, pad_idx=0):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim, padding_idx=pad_idx)

        self.rnn = nn.LSTM(
            input_size=emb_dim,
            hidden_size=hid_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,  # IMPORTANT
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_lengths):
        # src: [B, S]
        embedded = self.dropout(self.embedding(src))  # [B, S, E]

        # Dùng pack_padded_sequence để bỏ qua padding khi chạy LSTM
        packed = pack_padded_sequence(
            embedded,
            src_lengths.cpu(),
            batch_first=True,
            enforce_sorted=False
        )

        packed_outputs, (hidden, cell) = self.rnn(packed)

        # Khôi phục lại tensor có padding để dùng cho attention
        # encoder_outputs: [B, S, H] (đã restore padding)
        encoder_outputs, _ = pad_packed_sequence(packed_outputs, batch_first=True)

        return encoder_outputs, (hidden, cell)


# =========================
# Decoder (no attention)
# =========================
class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers=1, dropout=0.0, pad_idx=0):
        super().__init__()
        self.output_dim = output_dim

        self.embedding = nn.Embedding(output_dim, emb_dim, padding_idx=pad_idx)

        self.rnn = nn.LSTM(
            input_size=emb_dim,
            hidden_size=hid_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,  # IMPORTANT
        )

        self.fc_out = nn.Linear(hid_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_step, hidden, cell):
        """
        input_step: [B, 1] (token id)
        hidden, cell: [n_layers, B, H]
        """
        embedded = self.dropout(self.embedding(input_step))  # [B, 1, E]
        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))  # output: [B, 1, H]
        pred = self.fc_out(output.squeeze(1))  # [B, V]
        return pred, hidden, cell


# =========================
# Seq2Seq (no attention)
# =========================
class Seq2Seq(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, src_len, tgt_in, teacher_forcing_ratio=0.5):
        """
        src: [B, S]
        src_len: [B]
        tgt_in: [B, T]
        outputs: [B, T, V]
        """
        B = src.size(0)
        T = tgt_in.size(1)
        V = self.decoder.output_dim

        outputs = torch.zeros(B, T, V, device=self.device)

        # Encoder
        _, (hidden, cell) = self.encoder(src, src_len)

        # First input token = <sos>
        input_tok = tgt_in[:, 0].unsqueeze(1)  # [B, 1]

        # Decoder sinh từng token một theo thời gian
        # Có sử dụng teacher forcing với xác suất teacher_forcing_ratio
        for t in range(1, T):
            pred, hidden, cell = self.decoder(input_tok, hidden, cell)
            outputs[:, t] = pred

            teacher_force = random.random() < teacher_forcing_ratio
            top1 = pred.argmax(1).unsqueeze(1)  # [B, 1]
            input_tok = tgt_in[:, t].unsqueeze(1) if teacher_force else top1

        return outputs

    # Greedy decoding: tại mỗi bước chọn token có xác suất cao nhất
    def greedy_decode(self, src, src_len, max_len=50, sos_idx=None, eos_idx=None):
        """
        Return: LongTensor [B, <=max_len]
        """
        assert sos_idx is not None and eos_idx is not None

        B = src.size(0)
        _, (hidden, cell) = self.encoder(src, src_len)

        input_tok = torch.full((B, 1), sos_idx, dtype=torch.long, device=self.device)

        outputs = []
        finished = torch.zeros(B, dtype=torch.bool, device=self.device)

        for _ in range(max_len):
            pred, hidden, cell = self.decoder(input_tok, hidden, cell)
            top1 = pred.argmax(1)  # [B]
            outputs.append(top1)

            finished |= (top1 == eos_idx)
            if finished.all():
                break

            input_tok = top1.unsqueeze(1)

        return torch.stack(outputs, dim=1)  # [B, L]


# =========================
# Luong Attention
# =========================
class LuongAttention(nn.Module):
    def __init__(self, hid_dim):
        super().__init__()
        self.hid_dim = hid_dim

    def forward(self, decoder_hidden, encoder_outputs, mask=None):
        """
        decoder_hidden: [B, H]
        encoder_outputs: [B, S, H]
        mask: [B, S] optional (1=keep, 0=mask)
        """
        # Tính attention score bằng dot-product giữa:
        # encoder_outputs (B,S,H) và decoder_hidden (B,H)
        # dot: (B,S,H) x (B,H,1) -> (B,S,1) -> (B,S)
        energy = torch.bmm(encoder_outputs, decoder_hidden.unsqueeze(2)).squeeze(2)  # [B, S]

        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)

        # Softmax theo chiều S để thu được phân phối attention
        attn_weights = F.softmax(energy, dim=1)  # [B, S]

        # Context vector = tổng có trọng số của encoder outputs
        # context: (B,1,S) x (B,S,H) -> (B,1,H) -> (B,H)
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs).squeeze(1)  # [B, H]

        return context, attn_weights


# =========================
# Attention Decoder
# =========================
# Attention Decoder:
# 1. Embed input token
# 2. Tính attention để lấy context
# 3. Ghép embedding + context đưa vào LSTM
# 4. Dự đoán token tiếp theo
class AttentionDecoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers=1, dropout=0.3, pad_idx=0):
        super().__init__()

        self.output_dim = output_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers

        self.embedding = nn.Embedding(output_dim, emb_dim, padding_idx=pad_idx)
        self.attention = LuongAttention(hid_dim)

        self.rnn = nn.LSTM(
            input_size=emb_dim + hid_dim,
            hidden_size=hid_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,  # IMPORTANT
        )

        self.fc_out = nn.Linear(hid_dim * 2, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_tok, hidden, cell, encoder_outputs, mask=None):
        """
        input_tok: [B] (token id)
        hidden, cell: [n_layers, B, H]
        encoder_outputs: [B, S, H]
        """
        input_tok = input_tok.unsqueeze(1)  # [B, 1]
        emb = self.dropout(self.embedding(input_tok))  # [B, 1, E]

        # attention dùng hidden layer cuối
        context, attn_weights = self.attention(hidden[-1], encoder_outputs, mask)  # context: [B,H]
        context = context.unsqueeze(1)  # [B, 1, H]

        rnn_input = torch.cat((emb, context), dim=2)  # [B,1,E+H]
        output, (hidden, cell) = self.rnn(rnn_input, (hidden, cell))  # output: [B,1,H]

        output = output.squeeze(1)   # [B,H]
        context = context.squeeze(1) # [B,H]

        pred = self.fc_out(torch.cat((output, context), dim=1))  # [B,V]
        return pred, hidden, cell, attn_weights


# =========================
# Seq2Seq with Attention
# =========================
class Seq2SeqWithAttention(nn.Module):
    def __init__(self, encoder: Encoder, decoder: AttentionDecoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, src_lens, tgt, teacher_forcing_ratio=0.5):
        """
        src: [B,S], tgt: [B,T]
        outputs: [B,T,V]
        """
        B, T = tgt.size()
        V = self.decoder.output_dim
        outputs = torch.zeros(B, T, V, device=self.device)

        encoder_outputs, (hidden, cell) = self.encoder(src, src_lens)

        input_tok = tgt[:, 0]  # [B] = <sos>

        for t in range(1, T):
            pred, hidden, cell, _ = self.decoder(input_tok, hidden, cell, encoder_outputs)
            outputs[:, t] = pred

            teacher_force = random.random() < teacher_forcing_ratio
            top1 = pred.argmax(1)  # [B]
            input_tok = tgt[:, t] if teacher_force else top1

        return outputs


    # Greedy decoding: tại mỗi bước chọn token có xác suất cao nhất
    def greedy_decode(self, src, src_lens, max_len, sos_idx, eos_idx):
        """
        Return: LongTensor [B, <=max_len] (dừng khi tất cả gặp eos)
        """
        self.eval()
        encoder_outputs, (hidden, cell) = self.encoder(src, src_lens)

        B = src.size(0)
        input_tok = torch.full((B,), sos_idx, dtype=torch.long, device=self.device)

        outputs = []
        finished = torch.zeros(B, dtype=torch.bool, device=self.device)

        for _ in range(max_len):
            pred, hidden, cell, _ = self.decoder(input_tok, hidden, cell, encoder_outputs)
            top1 = pred.argmax(1)  # [B]
            outputs.append(top1)

            finished |= (top1 == eos_idx)
            if finished.all():
                break

            input_tok = top1

        return torch.stack(outputs, dim=1)  # [B, L]