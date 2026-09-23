"""Mask-aware sequence encoder for the first observed packets."""

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence


class TemporalBiGRU(nn.Module):
    def __init__(self, input_features=3, hidden_size=64):
        super().__init__()
        if input_features not in (2, 3):
            raise ValueError('Use length+direction, optionally IAT')
        self.input_features = input_features
        self.gru = nn.GRU(input_features, hidden_size, batch_first=True, bidirectional=True)

    def forward(self, sequence, lengths):
        if sequence.ndim != 3 or sequence.size(-1) < self.input_features:
            raise ValueError('Expected [batch, packets, features]')
        if lengths.ndim != 1 or lengths.numel() != sequence.size(0):
            raise ValueError('Invalid sequence lengths')
        if torch.any(lengths < 1) or torch.any(lengths > sequence.size(1)):
            raise ValueError('Sequence length out of range')
        active = sequence[..., :self.input_features]
        if not torch.isfinite(active).all():
            raise ValueError('Nonfinite sequence feature')
        packed = pack_padded_sequence(active, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, hidden = self.gru(packed)
        return torch.cat((hidden[-2], hidden[-1]), dim=-1)
