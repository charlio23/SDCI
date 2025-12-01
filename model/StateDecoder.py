

from torch import nn
from model.modules_ACD import MLP


class StateDecoder(nn.Module):

    def __init__(self, n_in, n_hid, n_states, n_atoms, decoder_type='region'):
        super(StateDecoder, self).__init__()
        if decoder_type=='region':
            self.prior = MLP(n_in, n_hid, n_states, use_batch_norm=False, final_linear=True)
        else:
            self.prior = MLP(2*n_in+n_states, n_hid, n_states, use_batch_norm=False, final_linear=True)
    def forward(self, input):
        
        # input shape: [B, num_atoms, T, n_in | n_in*num_states]
        
        return self.prior(input).softmax(-1)
