#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""
import torch
### YOUR CODE HERE for part 1i
class CNN(torch.nn.Module):
    def __init__(self, e_char, kernel_size, filters, max_word_len):
        super(CNN, self).__init__()
        self.conv1d = torch.nn.Conv1d(in_channels=e_char,
                                      out_channels=filters,
                                      kernel_size=kernel_size,
                                      bias=True)
        self.relu = torch.nn.ReLU()
        self.maxpool = torch.nn.MaxPool1d(max_word_len - kernel_size + 1)

    def forward(self, input):
        return self.maxpool(self.relu(self.conv1d(input))).squeeze()

### END YOUR CODE

