#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch

"""
CS224N 2018-19: Homework 5
"""


### YOUR CODE HERE for part 1h
class Highway(torch.nn.Module):
    def __init__(self, word_emb_len, dropout_rate):
        """


        """
        super(Highway, self).__init__()
        self.proj = torch.nn.Linear(word_emb_len, word_emb_len, bias=True)
        self.gate = torch.nn.Linear(word_emb_len, word_emb_len, bias=True)
        self.relu = torch.nn.ReLU()
        self.sigma = torch.nn.Sigmoid()
        self.dropout = torch.nn.Dropout(dropout_rate)

    def forward(self, input):
        """

        :param input: tensor with shape=(max_sent_len, batch_size, word_emb_len)
        :return:
        """
        proj = self.proj(input)
        proj = self.relu(proj)
        gate = self.gate(input)
        gate = self.sigma(gate)
        highway = proj * gate + (1 - gate) * input
        word_emb = self.dropout(highway)
        return word_emb

### END YOUR CODE
