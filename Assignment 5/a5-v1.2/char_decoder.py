#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class CharDecoder(nn.Module):
    def __init__(self, hidden_size, char_embedding_size=50, target_vocab=None):
        """ Init Character Decoder.

        @param hidden_size (int): Hidden size of the decoder LSTM
        @param char_embedding_size (int): dimensionality of character embeddings
        @param target_vocab (VocabEntry): vocabulary for the target language. See vocab.py for documentation.
        """
        ### YOUR CODE HERE for part 2a
        ### TODO - Initialize as an nn.Module.
        ###      - Initialize the following variables:
        ###        self.charDecoder: LSTM. Please use nn.LSTM() to construct this.
        ###        self.char_output_projection: Linear layer, called W_{dec} and b_{dec} in the PDF
        ###        self.decoderCharEmb: Embedding matrix of character embeddings
        ###        self.target_vocab: vocabulary for the target language
        ###
        ### Hint: - Use target_vocab.char2id to access the character vocabulary for the target language.
        ###       - Set the padding_idx argument of the embedding matrix.
        ###       - Create a new Embedding layer. Do not reuse embeddings created in Part 1 of this assignment.
        super(CharDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.char_embedding_size = char_embedding_size
        self.target_vocab = target_vocab
        self.charDecoder = nn.LSTM(input_size=char_embedding_size,
                                   hidden_size=hidden_size)
        self.char_output_projection = nn.Linear(in_features=hidden_size,
                                                out_features=len(target_vocab.char2id))
        self.decoderCharEmb = nn.Embedding(num_embeddings=len(target_vocab.char2id),
                                           embedding_dim=char_embedding_size,
                                           padding_idx=target_vocab.char2id['<pad>'])

        ### END YOUR CODE

    def forward(self, input, dec_hidden=None):
        """ Forward pass of character decoder.

        @param input: tensor of integers, shape (length, batch)
        @param dec_hidden: internal state of the LSTM before reading the input characters. A tuple of two tensors of shape (1, batch, hidden_size)

        @returns scores: called s_t in the PDF, shape (length, batch, self.vocab_size)
        @returns dec_hidden: internal state of the LSTM after reading the input characters. A tuple of two tensors of shape (1, batch, hidden_size)
        """
        ### YOUR CODE HERE for part 2b
        ### TODO - Implement the forward pass of the character decoder.
        char_emb = self.decoderCharEmb(torch.t(input)).permute(1, 0, 2)
        lstm_out, dec_hidden = self.charDecoder(char_emb, dec_hidden)
        scores = self.char_output_projection(lstm_out)
        return scores, dec_hidden

        ### END YOUR CODE 

    def train_forward(self, char_sequence, dec_hidden=None):
        """ Forward computation during training.

        @param char_sequence: tensor of integers, shape (length, batch). Note that "length" here and in forward() need not be the same.
        @param dec_hidden: initial internal state of the LSTM, obtained from the output of the word-level decoder. A tuple of two tensors of shape (1, batch, hidden_size)

        @returns The cross-entropy loss, computed as the *sum* of cross-entropy losses of all the words in the batch.
        """
        ### YOUR CODE HERE for part 2c
        ### TODO - Implement training forward pass.
        ###
        ### Hint: - Make sure padding characters do not contribute to the cross-entropy loss.
        ###       - char_sequence corresponds to the sequence x_1 ... x_{n+1} from the handout (e.g., <START>,m,u,s,i,c,<END>).
        scores, _ = self.forward(char_sequence[:-1], dec_hidden)
        P = - F.log_softmax(scores, dim=-1)
        loss = torch.gather(P, index=char_sequence[1:].unsqueeze(-1), dim=-1).squeeze()
        loss = loss.sum()
        return loss

        ### END YOUR CODE

    def decode_greedy(self, initialStates, device, max_length=21):
        """ Greedy decoding
        @param initialStates: initial internal state of the LSTM, a tuple of two tensors of size (1, batch, hidden_size)
        @param device: torch.device (indicates whether the model is on CPU or GPU)
        @param max_length: maximum length of words to decode

        @returns decodedWords: a list (of length batch) of strings, each of which has length <= max_length.
                              The decoded strings should NOT contain the start-of-word and end-of-word characters.
        """

        ### YOUR CODE HERE for part 2d
        ### TODO - Implement greedy decoding.
        ### Hints:
        ###      - Use target_vocab.char2id and target_vocab.id2char to convert between integers and characters
        ###      - Use torch.tensor(..., device=device) to turn a list of character indices into a tensor.
        ###      - We use curly brackets as start-of-word and end-of-word characters. That is, use the character '{' for <START> and '}' for <END>.
        ###        Their indices are self.target_vocab.start_of_word and self.target_vocab.end_of_word, respectively.
        batch_size = initialStates[0].shape[1]
        input = [[self.target_vocab.start_of_word] for i in range(batch_size)]
        input = torch.t(torch.tensor(input, device=device, dtype=torch.long))

        hyp = [['{'] for i in range(batch_size)]

        dec_hidden = initialStates
        for i in range(max_length):
            scores, dec_hidden = self.forward(input, dec_hidden)
            P = (F.softmax(scores, dim=-1)).permute(1, 0, 2)
            _, index = torch.max(P, dim=-1)
            index = index.cpu().squeeze(dim=-1).numpy()
            for j in range(batch_size):
                if hyp[j][-1] != '}':
                    hyp[j].append(self.target_vocab.id2char[index[j]])
            index = np.expand_dims(index, axis=-1)
            input = torch.t(torch.tensor(index, device=device, dtype=torch.long))

        for i in range(batch_size):
            if hyp[i][-1] == '}':
                hyp[i] = ''.join(hyp[i][1: -1])
            else:
                hyp[i] = ''.join(hyp[i][1:])
        return hyp
        ### END YOUR CODE
