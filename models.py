import torch
from torch import nn
import torch.nn.functional as F
import torchvision
import sys
from st_gumbel import gumbel_softmax, st_gumbel_softmax

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TransformerModel(nn.Module):

    def __init__(self, embed_weight, d_model, nhead, dim_feedforward, num_encoder_layers, num_decoder_layers, seq_len, dropout=0.5):
        super(TransformerModel, self).__init__()
        self.model_type = 'Transformer'
        self.vocab_size = embed_weight.shape[0]-2
        self.dropout = nn.Dropout(p=dropout)

        self.embedding = nn.Embedding.from_pretrained(torch.from_numpy(embed_weight), freeze=True)
        self.enc_pos_encoding = torch.autograd.Variable(torch.empty((64, d_model), device='cuda').uniform_(-0.1, 0.1),
                                                        requires_grad=True)
        self.dec_pos_encoding = torch.autograd.Variable(torch.empty((seq_len, d_model), device='cuda').uniform_(-0.1, 0.1),
                                                    requires_grad=True)

        self.fc1 = nn.Linear(1024, d_model)
        self.transformer = nn.Transformer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
                                          num_encoder_layers=num_encoder_layers, num_decoder_layers=num_decoder_layers)
        self.fc2 = nn.Linear(d_model, self.vocab_size)

    def generate_pad_mask(self, batch_size, max_len, caption_lengths):

        # print(caption_lengths.size())
        # print(caption_lengths.type())
        mask = torch.full((batch_size, max_len), fill_value=True, dtype=torch.bool, device='cuda')
        # print(mask)
        for ind, cap_len in enumerate(caption_lengths):
            mask[ind][:cap_len] = False

        # print(mask)
        # print(mask.sum(dim=1))
        # print(caption_lengths)
        # sys.exit()
        return mask

    def forward(self, image, encoded_captions, caption_lengths):
        batch_size = image.size(0)
        max_len = encoded_captions.size(1)
        encoder_dim = image.size(-1)
        vocab_size = self.vocab_size

        embedded_captions = self.embedding(encoded_captions)  # (batch_size, max_caption_length, embed_dim)
        embedded_captions += self.dec_pos_encoding.narrow(0, 0, encoded_captions.size(1))

        # Flatten image
        image = image.view(batch_size, -1, encoder_dim)  # (batch_size, num_pixels, encoder_dim)

        image = self.fc1(image)
        image += self.enc_pos_encoding

        padding_mask = self.generate_pad_mask(batch_size, max_len, caption_lengths)

        tgt_mask = self.transformer.generate_square_subsequent_mask(max_len).to('cuda')

        # print(image.size())
        # print(embedded_captions.size())

        output = self.transformer(image.transpose(0, 1), embedded_captions.transpose(0, 1), tgt_mask=tgt_mask, tgt_key_padding_mask=padding_mask).transpose(0, 1)

        preds = self.fc2(self.dropout(output))
        # print(preds)
        # sys.exit()

        return preds


class TransformerGumbelModel(nn.Module):

    def __init__(self, embed_weight, transformer, chexpert, temperature, beta, dropout=0.5):
        super(TransformerGumbelModel, self).__init__()
        self.model_type = 'TransformerGB'
        self.vocab_size = embed_weight.shape[0] - 2
        self.dropout = nn.Dropout(p=dropout)
        self.temperature = temperature
        self.beta = beta

        self.embedding = transformer.embedding
        self.onehot_embedding = nn.Linear(embed_weight.shape[0], embed_weight.shape[1], bias=False)
        with torch.no_grad():
            self.onehot_embedding.weight = torch.nn.Parameter(torch.tensor(embed_weight.transpose())[:, :-2])
            for param in self.onehot_embedding.parameters():
                param.requires_grad = False
        self.enc_pos_encoding = transformer.enc_pos_encoding
        self.dec_pos_encoding = transformer.dec_pos_encoding

        self.fc1 = transformer.fc1
        self.transformer = transformer.transformer
        self.fc2 = transformer.fc2

        self.chexpert = chexpert
        for param in self.chexpert.parameters():
            param.requires_grad = False

    def generate_pad_mask(self, batch_size, max_len, caption_lengths):
        mask = torch.full((batch_size, max_len), fill_value=True, dtype=torch.bool, device='cuda')
        for ind, cap_len in enumerate(caption_lengths):
            mask[ind][:cap_len] = False
        return mask

    def set_temperature(self, temperature):
        self.temperature = temperature

    def apply_chexpert(self, embs, caption_lengths):
        x = embs
        batch_size = embs.size(0)
        max_len = embs.size(1)
        padding_mask = self.chexpert.generate_pad_mask(batch_size, max_len, caption_lengths)

        output, (_, _) = self.chexpert.rnn(x)

        y_hats = [attn(output, padding_mask) for attn in self.chexpert.attns]
        y_hats = torch.stack(y_hats, dim=1)

        return y_hats

    def forward(self, image, encoded_captions, caption_lengths, eval=False):
        batch_size = image.size(0)
        max_len = encoded_captions.size(1)
        encoder_dim = image.size(-1)
        vocab_size = self.vocab_size

        embedded_captions = self.embedding(encoded_captions)  # (batch_size, max_caption_length, embed_dim)
        embedded_captions += self.dec_pos_encoding.narrow(0, 0, encoded_captions.size(1))

        # Flatten image
        image = image.view(batch_size, -1, encoder_dim)  # (batch_size, num_pixels, encoder_dim)

        image = self.fc1(image)
        image += self.enc_pos_encoding

        padding_mask = self.generate_pad_mask(batch_size, max_len, caption_lengths)

        tgt_mask = self.transformer.generate_square_subsequent_mask(max_len).to('cuda')


        output = self.transformer(image.transpose(0, 1), embedded_captions.transpose(0, 1), tgt_mask=tgt_mask, tgt_key_padding_mask=padding_mask).transpose(0, 1)

        preds = self.fc2(self.dropout(output))

        if eval:
            return preds

        one_hot_preds = gumbel_softmax(F.log_softmax(preds, dim=1), self.temperature, self.beta)  # (batch_size_t, vocab_size)
        embedded_preds = self.onehot_embedding(one_hot_preds)  # (batch_size_t, emb_dim)
        chexpert_preds = self.apply_chexpert(embedded_preds, caption_lengths)

        return preds, chexpert_preds


class TanhAttention(nn.Module):
    def __init__(self, hidden_size, dropout=0.5, num_out=3):
        super(TanhAttention, self).__init__()
        self.attn1 = nn.Linear(hidden_size, hidden_size // 2)
        self.attn2 = nn.Linear(hidden_size // 2, 1, bias=False)
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(hidden_size, num_out)

    def forward(self, output, mask):
        attn1 = nn.Tanh()(self.attn1(output))
        # print(attn1.size())
        attn2 = self.attn2(attn1).squeeze(-1)
        # print(attn2.size())
        # print(mask.size())
        attn = F.softmax(torch.add(attn2, mask), dim=1)
        # print(attn.size())

        h = output.transpose(1, 2).matmul(attn.unsqueeze(2)).squeeze(2)
        # print(h.size())
        y_hat = self.fc(self.dropout(h))
        # print(y_hat.size())
        # sys.exit()

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
        # print(attn.size())
        # print(attn2.size())
        # print(mask.size())
        attn = F.softmax(torch.add(attn, mask), dim=1)
        # print(attn.size())

        h = output.transpose(1, 2).matmul(attn.unsqueeze(2)).squeeze(2)
        # print(h.size())
        y_hat = self.fc(self.dropout(h))
        # print(y_hat.size())
        # sys.exit()

        return y_hat

class LSTM_Attn(nn.Module):
    def __init__(self, embed_weight, emb_dim, hidden_size, num_classes=14):

        super(LSTM_Attn, self).__init__()

        self.embed = nn.Embedding.from_pretrained(torch.from_numpy(embed_weight), freeze=True)

        self.rnn = nn.LSTM(input_size=emb_dim, hidden_size=hidden_size, batch_first=True, bidirectional=True)

        self.attns = nn.ModuleList([TanhAttention(hidden_size*2) for i in range(num_classes)])

    def generate_pad_mask(self, batch_size, max_len, caption_lengths):

        # print(caption_lengths.size())
        # print(caption_lengths.type())
        mask = torch.full((batch_size, max_len), fill_value=float('-inf'), dtype=torch.float, device='cuda')
        # print(mask)
        for ind, cap_len in enumerate(caption_lengths):
            mask[ind][:cap_len] = 0

        # print(mask)
        # print(mask.sum(dim=1))
        # print(caption_lengths)
        # sys.exit()
        return mask

    def forward(self, encoded_captions, caption_lengths):
        x = self.embed(encoded_captions)
        # print(x.size())

        batch_size = encoded_captions.size(0)
        max_len = encoded_captions.size(1)
        padding_mask = self.generate_pad_mask(batch_size, max_len, caption_lengths)

        output, (_, _) = self.rnn(x)

        # print(output.size())

        y_hats = [attn(output, padding_mask) for attn in self.attns]
        # print(y_hats[0].size())
        y_hats = torch.stack(y_hats, dim=1)
        # print(y_hats.size())
        # sys.exit()

        return y_hats


class LSTM_Attn(nn.Module):
    def __init__(self, embed_weight, emb_dim, hidden_size, num_classes=14):

        super(LSTM_Attn, self).__init__()

        self.embed = nn.Embedding.from_pretrained(torch.from_numpy(embed_weight), freeze=True)

        self.rnn = nn.LSTM(input_size=emb_dim, hidden_size=hidden_size, batch_first=True, bidirectional=True)

        self.attns = nn.ModuleList([TanhAttention(hidden_size*2) for i in range(num_classes)])

    def generate_pad_mask(self, batch_size, max_len, caption_lengths):

        # print(caption_lengths.size())
        # print(caption_lengths.type())
        mask = torch.full((batch_size, max_len), fill_value=float('-inf'), dtype=torch.float, device='cuda')
        # print(mask)
        for ind, cap_len in enumerate(caption_lengths):
            mask[ind][:cap_len] = 0

        # print(mask)
        # print(mask.sum(dim=1))
        # print(caption_lengths)
        # sys.exit()
        return mask

    def forward(self, encoded_captions, caption_lengths):
        x = self.embed(encoded_captions)
        # print(x.size())

        batch_size = encoded_captions.size(0)
        max_len = encoded_captions.size(1)
        padding_mask = self.generate_pad_mask(batch_size, max_len, caption_lengths)

        output, (_, _) = self.rnn(x)

        # print(output.size())

        y_hats = [attn(output, padding_mask) for attn in self.attns]
        # print(y_hats[0].size())
        y_hats = torch.stack(y_hats, dim=1)
        # print(y_hats.size())
        # sys.exit()

        return y_hats

class CNN_Attn(nn.Module):
    def __init__(self, embed_weight, emb_dim, filters, kernels, num_classes=14):

        super(CNN_Attn, self).__init__()

        self.embed = nn.Embedding.from_pretrained(torch.from_numpy(embed_weight), freeze=True)

        self.Ks = kernels

        self.convs = nn.ModuleList([nn.Conv1d(emb_dim, filters, K) for K in self.Ks])

        self.attns = nn.ModuleList([DotAttention(filters) for _ in range(num_classes)])

    def generate_pad_mask(self, batch_size, max_len, caption_lengths):

        # print(caption_lengths.size())
        # print(caption_lengths.type())
        total_len = max_len*len(self.Ks)
        for K in self.Ks:
            total_len -= (K-1)
        mask = torch.full((batch_size, total_len), fill_value=float('-inf'), dtype=torch.float, device='cuda')
        # print(mask)
        for ind1, cap_len in enumerate(caption_lengths):
            for ind2, K in enumerate(self.Ks):
                mask[ind1][max_len*ind2:cap_len-(K-1)] = 0

        # print(mask)
        # print(mask.sum(dim=1))
        # print(caption_lengths)
        # sys.exit()
        return mask

    def forward(self, encoded_captions, caption_lengths):
        x = self.embed(encoded_captions).transpose(1, 2)
        # print(x.size())

        batch_size = encoded_captions.size(0)
        max_len = encoded_captions.size(1)
        padding_mask = self.generate_pad_mask(batch_size, max_len, caption_lengths)

        output = [F.relu(conv(x)).transpose(1, 2) for conv in self.convs]
        output = torch.cat(output, dim=1)


        # print(output.size())

        y_hats = [attn(output, padding_mask) for attn in self.attns]
        # print(y_hats[0].size())
        y_hats = torch.stack(y_hats, dim=1)
        # print(y_hats.size())
        # sys.exit()

        return y_hats
