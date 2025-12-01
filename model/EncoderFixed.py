import torch
from torch import nn

class EncoderFixed(nn.Module):
    """
    Reduced architecture for dynamics decoder
    """
    def __init__(self, n_in, n_edge_types, n_states):
        super(EncoderFixed, self).__init__()

        self.U = nn.Parameter(torch.randn(n_in*(n_in-1), n_states, n_edge_types))

    def forward(self):

        return self.U - self.U.logsumexp(-1, keepdim=True)