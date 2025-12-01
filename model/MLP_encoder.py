"""
Code extracted from https://github.com/loeweX/AmortizedCausalDiscovery (MIT License).
"""

import torch

from model.modules_ACD import *
from model.encoder_ACD import Encoder


class MLPEncoder(Encoder):
    """Based on https://github.com/ethanfetaya/NRI (MIT License)."""
    """Same MLP encoder as ACD, but this time we output n_edge_types*n_states"""

    def __init__(self, n_in, n_hid, n_edge_types, n_states, do_prob=0.0, factor=True):
        super().__init__(factor)

        self.n_edge_types = n_edge_types
        self.n_states = n_states

        self.mlp1 = MLP(n_in, n_hid, n_hid, do_prob)
        self.mlp2 = MLP(n_hid * 2, n_hid, n_hid, do_prob)
        self.mlp3 = MLP(n_hid, n_hid, n_hid, do_prob)
        if self.factor:
            self.mlp4 = MLP(n_hid * 3, n_hid, n_hid, do_prob)
            print("Using factor graph MLP encoder.")
        else:
            self.mlp4 = MLP(n_hid * 2, n_hid, n_hid, do_prob)
            print("Using MLP encoder.")
        self.fc_out = nn.Linear(n_hid, n_edge_types*n_states)

        self.init_weights()

    def forward(self, inputs, rel_rec, rel_send):
        # Input shape: [num_sims, num_atoms, num_timesteps, num_dims]
        x = inputs.reshape(inputs.size(0), inputs.size(1), -1)
        # New shape: [num_sims, num_atoms, num_timesteps*num_dims]
        x = self.mlp1(x)  # 2-layer ELU net per node
        x = self.node2edge(x, rel_rec, rel_send)
        x = self.mlp2(x)
        x_skip = x

        if self.factor:
            x = self.edge2node(x, rel_rec, rel_send)
            x = self.mlp3(x)
            x = self.node2edge(x, rel_rec, rel_send)
            x = torch.cat((x, x_skip), dim=2)  # Skip connection
            x = self.mlp4(x)
        else:
            x = self.mlp3(x)
            x = torch.cat((x, x_skip), dim=2)  # Skip connection
            x = self.mlp4(x)
        batch, num_rel, _ = x.size()
        return self.fc_out(x).reshape(batch, num_rel, self.n_states, self.n_edge_types)

class StateEncoderRegion(nn.Module):

    def __init__(self, n_in, n_hid, n_states):
        super(StateEncoderRegion, self).__init__()
        self.mlp1 = MLP(n_in, n_hid, n_hid, use_batch_norm=False, final_linear=False, activation='elu')
        self.out = nn.Linear(n_hid, n_states)
    def forward(self, input):
        
        # input shape: [B, num_atoms, T, n_in]
        return self.out(self.mlp1(input))
        #return self.mlp1(input)
