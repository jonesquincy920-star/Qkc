import torch
import torch.nn as nn

from .topology import NUM_NODES, NODE_DIM


class EnneractBrain(nn.Module):
    """The Orrery Engine: propagates signals across the 9D hypercube graph."""

    def __init__(self):
        super().__init__()
        self.transform = nn.Linear(NODE_DIM, NODE_DIM)
        self.activation = nn.Tanh()

    def forward(self, x, adjacency):
        neighbor_info = torch.matmul(adjacency, x)
        return self.activation(self.transform(neighbor_info + x))


def encode_to_brain(data_vector, shells, device):
    """Maps a 512-element data vector onto brain nodes, weighted by shell depth."""
    brain_input = torch.zeros((NUM_NODES, NODE_DIM)).to(device)
    for i in range(NUM_NODES):
        brain_input[i] = data_vector[i] * (shells[i] + 1)
    return brain_input.unsqueeze(0)
