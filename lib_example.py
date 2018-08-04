import torch
import torch.nn as nn

# import onmt
# import onmt.io
# from .onmt import io
import onmt.modules

# We begin by loading in the vocabulary for the model of interest

vocab = dict(torch.load("data/data.vocab.pt"))
# from onmt.inputters.dataset_base import (DatasetBase, UNK_WORD, PAD_WORD, BOS_WORD, EOS_WORD)
src_padding = vocab["src"].stoi[onmt.inputters.dataset_base.PAD_WORD]
tgt_padding = vocab["tgt"].stoi[onmt.inputters.dataset_base.PAD_WORD]

# src_padding = vocab["src"].stoi[onmt.io.PAD_WORD]
# tgt_padding = vocab["tgt"].stoi[onmt.io.PAD_WORD]

emb_size = 10
rnn_size = 6

# Specify the core model.
encoder_embeddings = onmt.modules.Embeddings(emb_size, len(vocab["src"]),
                                             word_padding_idx=src_padding)
encoder = onmt.encoders.RNNEncoder(hidden_size=rnn_size, num_layers=1,
                                  rnn_type="LSTM", bidirectional=True,
                                  embeddings=encoder_embeddings)

decoder_embeddings = onmt.modules.Embeddings(emb_size, len(vocab["tgt"]),
                                             word_padding_idx=tgt_padding)

from onmt.decoders.decoder import InputFeedRNNDecoder as InputFeedRNNDecoder

decoder = InputFeedRNNDecoder(hidden_size=rnn_size, num_layers=1,
                                           bidirectional_encoder=True,
                                           rnn_type="LSTM", embeddings=decoder_embeddings)

# from onmt.models.model import NMTModel as NMTModel
model = onmt.models.model.NMTModel(encoder, decoder)

# Specify the tgt word generator and loss computation module
model.generator = nn.Sequential(
    nn.Linear(rnn_size, len(vocab["tgt"])),
    nn.LogSoftmax())

loss = onmt.utils.loss.NMTLossCompute(model.generator, vocab["tgt"])

# up the optimizer

optim = onmt.utils.optimizers.Optim(method="sgd", learning_rate=1, max_grad_norm=2)

optim.set_parameters(model.named_parameters())

# Load some data
data = torch.load("../../data/data.train.pt")
valid_data = torch.load("../../data/data.valid.pt")
data.load_fields(vocab)
valid_data.load_fields(vocab)
data.examples = data.examples[:100]

# To iterate through the data itself we use a torchtext iterator class. We specify one for both the training and test data.

train_iter = onmt.io.OrderedIterator(
    dataset=data, batch_size=10,
    device=-1,
    repeat=False)
valid_iter = onmt.io.OrderedIterator(
    dataset=valid_data, batch_size=10,
    device=-1,
    train=False)

trainer = onmt.Trainer(model, train_iter, valid_iter, loss, loss, optim)


def report_func(*args):
    stats = args[-1]
    stats.output(args[0], args[1], 10, 0)
    return stats


for epoch in range(2):
    trainer.train(epoch, report_func)
    val_stats = trainer.validate()

    print("Validation")
    val_stats.output(epoch, 11, 10, 0)
    trainer.epoch_step(val_stats.ppl(), epoch)


