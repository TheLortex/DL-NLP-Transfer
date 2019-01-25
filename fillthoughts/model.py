"""
This file implements the Fill-Thought architecture.
Taken and modified from https://github.com/sanyam5/skip-thoughts
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from config import *


class Encoder(nn.Module):
    thought_size = 1200
    word_size = 620

    @staticmethod
    def reverse_variable(var):
        idx = [i for i in range(var.size(0) - 1, -1, -1)]
        idx = Variable(torch.LongTensor(idx))

        if USE_CUDA:
            idx = idx.cuda(CUDA_DEVICE)

        inverted_var = var.index_select(0, idx)
        return inverted_var

    def __init__(self):
        super().__init__()
        
        self.word2embd = nn.Embedding(VOCAB_SIZE, self.word_size)
        self.lstm = nn.LSTM(self.word_size, self.thought_size, bidirectional=True)
        self.lstm_next = nn.LSTM(self.word_size, self.thought_size, bidirectional=True)

    def forward(self, sentences, category=None):
        # category = (n_categories)
        # sentences = (batch_size, maxlen), with padding on the right.
        sentences = sentences.transpose(0, 1)  # (maxlen, batch_size)

        word_embeddings = F.tanh(self.word2embd(sentences))  # (maxlen, batch_size, word_size)
            
        # The following is a hack: We read embeddings in reverse. This is required to move padding to the left.
        # If reversing is not done then the RNN sees a lot a garbage values right before its final state.
        # This reversing also means that the words will be read in reverse. But this is not a big problem since
        # several sequence to sequence models for Machine Translation do similar hacks.
        rev = self.reverse_variable(word_embeddings)

        _, (thoughts, _) = self.lstm(rev)
        thoughts = thoughts[-1]  # (batch, thought_size)

        return thoughts, word_embeddings


class SkipDecoder(nn.Module):

    word_size = Encoder.word_size

    def __init__(self, n_categories=0):
        super().__init__()
        self.lstm = nn.LSTM(2*Encoder.thought_size + self.word_size, self.word_size)
        self.worder = nn.Linear(self.word_size, VOCAB_SIZE)

    def forward(self, thoughts, word_embeddings=None, word2embed=None):
        # thoughts = (batch_size, Encoder.thought_size)
        # word_embeddings = # (maxlen, batch, word_size)
        batch_size, _ = thoughts.shape

        # We need to provide the current sentences's embedding or "thought" at every timestep.
        thoughts = thoughts.repeat(MAXLEN, 1, 1)  # (maxlen, batch, thought_size)
        thoughts_prev = thoughts[:, :-2, :] # (maxlen, batch-2, thought_size)
        thoughts_next = thoughts[:, 2:, :] # Offset by two.
        
        thoughts_input = torch.cat([thoughts_prev, thoughts_next], dim=2) # (maxlen, batch-2, 2*thought_size)
        
       
        if word_embeddings is None:
            last_word = torch.zeros([1, batch_size-2, self.word_size], dtype=torch.float32, device=CUDA_DEVICE)
            result = []
            for _ in range(MAXLEN):
                lstm_input = torch.cat([thoughts_input[0].unsqueeze(dim=0), last_word], dim=2)
                pred_word, _ = self.lstm(lstm_input)
                last_word = pred_word.squeeze(dim=0) # (batch-2, word_size)
                last_word = self.worder(last_word) # (batch-2, vocab_size)
                _, last_word = last_word.max(dim=1)
                last_word = word2embed(last_word.unsqueeze(dim=0)) # (1, batch-2, word_size)
                
                result.append(pred_word)
            pred_embds = torch.cat(result, dim=0) # (maxlen, batch-2, word_size)
                
        else:
            # Teacher Forcing.
            #   1.) Prepare Word embeddings for the decoder.
            target_word_embeddings = word_embeddings[:, 1:-1, :]  # (maxlen, batch-2, word_size)

            #   2.) delay the embeddings by one timestep
            delayed_target_word_embeddings = torch.cat([0 * target_word_embeddings[-1:, :, :], target_word_embeddings[:-1, :, :]])

            # Supply current "thought" and delayed word embeddings for teacher forcing.
            pred_embds, _ = self.lstm(torch.cat([thoughts_input, delayed_target_word_embeddings], dim=2))  # (maxlen, batch-2, embd_size)

        # predict actual words
        a, b, c = pred_embds.size()
        pred = self.worder(pred_embds.view(a*b, c)).view(a, b, -1)  # (maxlen, batch-2, VOCAB_SIZE)
        
        pred = pred.transpose(0, 1).contiguous()  # (batch-2, maxlen, VOCAB_SIZE)

        return pred


class FillThoughts(nn.Module):

    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = SkipDecoder()

    def create_mask(self, var, lengths):
        mask = var.data.new().resize_as_(var.data).fill_(0)
#         print("lengths", lengths)
        for i, l in enumerate(lengths):
            for j in range(l):
                mask[i, j] = 1
        
        mask = Variable(mask)
        if USE_CUDA:
            mask = mask.cuda(var.get_device())
            
        return mask

    def forward(self, sentences, lengths):
        # cat_in = cat_out = (n_categories)
        # sentences = (B, maxlen)
        # lengths = (B)

        # Compute Thought Vectors for each sentence. Also get the actual word embeddings for teacher forcing.
        thoughts, word_embeddings = self.encoder(sentences)  # thoughts = (B, thought_size), word_embeddings = (B, maxlen, word_size)

        # Predict the words for the sentences
        pred = self.decoder(thoughts, word_embeddings)  # both = (batch-2, maxlen, VOCAB_SIZE)

        # mask the predictions, so that loss for beyond-EOS word predictions is cancelled.
        if MASK_BEYOND_EOS:
            mask = self.create_mask(pred, lengths[:-1])
            masked_pred = pred * mask
            pred = masked_pred
        
        loss = F.cross_entropy(pred.view(-1, VOCAB_SIZE), sentences[1:-1, :].view(-1))
        
        _, pred_ids = pred[0].max(1)

        return loss, sentences[1], pred_ids




