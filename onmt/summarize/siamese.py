import torch
import torch.nn as nn
import torch.nn.functional as F


# from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class DummyDiscriminator(nn.Module):
    def __init__(self):
        super(DummyDiscriminator, self).__init__()
        self.params = nn.Linear(2, 2)

    def forward(self, dec_output, enc_output):
        return 1.

    def _compute_loss_generator(self, enc_output, decoder_output):
        return 0


class SiameseDiscriminator(nn.Module):
    def __init__(self, decoder_dim, hidden_dim, bi=False, num_layers=1, batch_first=False, dropout=0.2):
        super(SiameseDiscriminator, self).__init__()
        self.hidden_dim = hidden_dim
        self.batch_first = batch_first
        self.drop_en = nn.Dropout(p=dropout)
        self.bi = bi
        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.rnn = nn.GRU(decoder_dim, hidden_dim, bidirectional=bi,
                          num_layers=num_layers, batch_first=batch_first, dropout=dropout)

        self.bn2 = nn.BatchNorm1d(hidden_dim * 2 * (2 if bi else 1))
        self.fc = nn.Linear(hidden_dim * 2 * (2 if bi else 1), 1)

    def forward(self, dec_output, enc_output):
        dec = self.seq2vec(dec_output)
        enc = self.seq2vec(enc_output)
        fc_input = self.bn2(torch.cat([enc, dec], dim=-1).squeeze(0))
        out = self.fc(fc_input)
        return out

    def seq2vec(self, x):
        '''
        Args:
            x: (time_step, batch,  input_size)
        Returns:
            num_output size
            
        from: https://github.com/keishinkickback/Pytorch-RNN-text-classification/blob/master/model.py
        '''

        x_embed = self.drop_en(x)

        # no need since this is outputted by the model
        # packed_input = pack_padded_sequence(x_embed, seq_lengths.cpu().numpy(),batch_first=True)

        # r_out shape (batch, time_step, output_size)
        # None is for initial hidden state
        packed_output, ht = self.rnn(x_embed, None)
        # out_rnn, _ = pad_packed_sequence(packed_output, batch_first=self.batch_first)

        # row_indices = torch.arange(0, x.size(0)).long()
        # col_indices = seq_lengths - 1
        # if next(self.parameters()).is_cuda:
        #     row_indices = row_indices.cuda()
        #     col_indices = col_indices.cuda()

        if self.bi:
            # h_n.view(num_layers, num_directions, batch, hidden_size)
            last_tensor = torch.cat([ht[0, :, :], packed_output[1, :, :]], dim=0)
        else:
            last_tensor = ht
        return last_tensor

    def _compute_loss_generator(self, enc_output, decoder_output):
        y_hat = self.forward(decoder_output, enc_output)
        # loss = F.binary_cross_entropy_with_logits(y_hat, 1)
        loss = F.binary_cross_entropy_with_logits(y_hat,
                                                  torch.ones(y_hat.size(), requires_grad=False).to(device=y_hat.device))
        return loss
