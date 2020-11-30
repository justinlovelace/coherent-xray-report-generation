import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn import metrics
import pdb
import pickle
import math
from torch.nn.init import xavier_uniform
import sys
import os


class TanhAttention(nn.Module):
    def __init__(self, hidden_size, dropout=0.5, num_out=3):
        super(TanhAttention, self).__init__()
        self.attn1 = nn.Linear(hidden_size, hidden_size // 2)
        self.attn2 = nn.Linear(hidden_size // 2, 1, bias=False)
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(hidden_size, num_out)

    def forward(self, output, mask):
        attn1 = nn.Tanh()(self.attn1(output))
        attn2 = self.attn2(attn1).squeeze(-1)
        attn = F.softmax(torch.add(attn2, mask), dim=1)

        h = output.transpose(1, 2).matmul(attn.unsqueeze(2)).squeeze(2)
        y_hat = self.fc(self.dropout(h))

        return y_hat

class DotAttention(nn.Module):
    def __init__(self, hidden_size, dropout=0.5, num_out=3):
        super(DotAttention, self).__init__()
        self.hidden_size = hidden_size
        self.attn = nn.Linear(hidden_size, 1, bias=False)
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(hidden_size, num_out)

    def forward(self, output, mask):
        attn = (self.attn(output) / (self.hidden_size ** 0.5)).squeeze(-1)
        attn = F.softmax(torch.add(attn, mask), dim=1)

        h = output.transpose(1, 2).matmul(attn.unsqueeze(2)).squeeze(2)
        y_hat = self.fc(self.dropout(h))

        return y_hat


class LSTM_Attn(nn.Module):
    def __init__(self, embed_weight, emb_dim, hidden_size, num_classes=14):

        super(LSTM_Attn, self).__init__()

        self.embed = nn.Embedding.from_pretrained(torch.from_numpy(embed_weight), freeze=True)

        self.rnn = nn.LSTM(input_size=emb_dim, hidden_size=hidden_size, batch_first=True, bidirectional=True)

        self.attns = nn.ModuleList([TanhAttention(hidden_size*2) for i in range(num_classes)])

    def generate_pad_mask(self, batch_size, max_len, caption_lengths):

        mask = torch.full((batch_size, max_len), fill_value=float('-inf'), dtype=torch.float, device='cuda')
        for ind, cap_len in enumerate(caption_lengths):
            mask[ind][:cap_len] = 0

        return mask

    def forward(self, encoded_captions, caption_lengths):
        x = self.embed(encoded_captions)

        batch_size = encoded_captions.size(0)
        max_len = encoded_captions.size(1)
        padding_mask = self.generate_pad_mask(batch_size, max_len, caption_lengths)

        output, (_, _) = self.rnn(x)


        y_hats = [attn(output, padding_mask) for attn in self.attns]
        y_hats = torch.stack(y_hats, dim=1)

        return y_hats

class CNN_Attn(nn.Module):
    def __init__(self, embed_weight, emb_dim, filters, kernels, num_classes=14):

        super(CNN_Attn, self).__init__()

        self.embed = nn.Embedding.from_pretrained(torch.from_numpy(embed_weight), freeze=True)

        self.Ks = kernels

        self.convs = nn.ModuleList([nn.Conv1d(emb_dim, filters, K) for K in self.Ks])

        self.attns = nn.ModuleList([DotAttention(filters) for _ in range(num_classes)])

    def generate_pad_mask(self, batch_size, max_len, caption_lengths):

        total_len = max_len*len(self.Ks)
        for K in self.Ks:
            total_len -= (K-1)
        mask = torch.full((batch_size, total_len), fill_value=float('-inf'), dtype=torch.float, device='cuda')
        for ind1, cap_len in enumerate(caption_lengths):
            for ind2, K in enumerate(self.Ks):
                mask[ind1][max_len*ind2:cap_len-(K-1)] = 0

        return mask

    def forward(self, encoded_captions, caption_lengths):
        x = self.embed(encoded_captions).transpose(1, 2)

        batch_size = encoded_captions.size(0)
        max_len = encoded_captions.size(1)
        padding_mask = self.generate_pad_mask(batch_size, max_len, caption_lengths)

        output = [F.relu(conv(x)).transpose(1, 2) for conv in self.convs]
        output = torch.cat(output, dim=1)


        y_hats = [attn(output, padding_mask) for attn in self.attns]
        y_hats = torch.stack(y_hats, dim=1)

        return y_hats
