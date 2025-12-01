
import torch
from model.modules_ACD import MLP
from torch import nn

class StateGRNNEncoder(nn.Module):
    """Based on https://github.com/ethanfetaya/NRI (MIT License)."""

    def __init__(
        self, n_in, n_hid, n_states, rnn='lstm', do_prob=0.0, factor=False
    ):
        super(StateGRNNEncoder, self).__init__()

        self.hidden_dim = n_hid
        self.n_states = n_states
        self.num_rec_layers = 2

        self.mlp1 = MLP(n_in, n_hid, n_hid, do_prob)
        self.mlp2 = MLP(n_hid*2, n_hid, n_hid, do_prob)
        self.mlp3f = MLP(n_hid, n_hid, n_hid, do_prob)
        #self.mlp4f = MLP(n_hid*2, n_hid, n_hid, do_prob)
        self.linearf = nn.Linear(n_hid*2, n_hid)

        self.mlp3b = MLP(n_hid, n_hid, n_hid, do_prob)
        #self.mlp4b = MLP(n_hid*2, n_hid, n_hid, do_prob)
        self.linearb = nn.Linear(n_hid*2, n_hid)
        

        if rnn == 'lstm':
            self.forward_hidden_decoder = nn.ModuleList([nn.LSTMCell(self.hidden_dim, self.hidden_dim)])
            for _ in range(1, self.num_rec_layers):
                self.forward_hidden_decoder.append(nn.LSTMCell(self.hidden_dim*2, self.hidden_dim))
            self.backward_hidden_decoder = nn.ModuleList([nn.LSTMCell(self.hidden_dim, self.hidden_dim)])
            for _ in range(1, self.num_rec_layers):
                self.backward_hidden_decoder.append(nn.LSTMCell(self.hidden_dim*2, self.hidden_dim))
        elif rnn == 'gru':
            self.rnn = nn.GRU(n_hid, n_hid, num_layers=3, batch_first=True)
        elif rnn == 'lru':
            raise NotImplementedError(' LRU not implemented')
            #self.rnn = LRU(n_hid, n_hid, n_hid)
        else:
            raise NotImplementedError(rnn + ' is not implemented as a RNN block.')
        self.mlp_out = MLP(n_hid*2, n_hid, n_states, do_prob)
        
        print("Using factor graph GRNN encoder.")

        self.init_weights()
    
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_normal(m.weight.data)
                m.bias.data.fill_(0.1)

    def edge2node(self, x, rel_rec):
        incoming = torch.matmul(rel_rec.t(), x)
        return incoming / incoming.size(1)

    def node2edge(self, x, rel_rec, rel_send):
        # filter the hidden representation of each timestep to consider only the information of the receiver/sender node
        # rel_rec[num_features * num_atoms, num_atoms] * x[num_samples, num_atoms, num_timesteps]
        # receivers, senders: (num_samples, num_atoms,  num_timesteps*num_dims)
        receivers = torch.matmul(rel_rec, x) 
        senders = torch.matmul(rel_send, x)

        # concatecate filtered reciever/sender hidden representation
        edges = torch.cat([senders, receivers], dim=2)
        return edges

    def forward(self, inputs, rel_rec, rel_send):

        # Input has shape: [num_sims, num_atoms, num_timesteps, num_dims]
        B, N, T, D = inputs.size()
        R = N*(N-1)
        x = inputs.transpose(1,2).reshape(B*T, N, D)
        # New shape: [num_sims*num_timesteps, num_atoms, num_dims]
        x = self.mlp1(x)  # 2-layer ELU net per node
        x = self.node2edge(x, rel_rec, rel_send)
        x = self.mlp2(x).reshape(B,T,R,self.hidden_dim)
        h_prev_fwd = [torch.zeros((B*R, self.hidden_dim)).to(inputs.device) for _ in range(self.num_rec_layers)]
        c_prev_fwd = [torch.zeros((B*R, self.hidden_dim)).to(inputs.device) for _ in range(self.num_rec_layers)]
        out_fwd = torch.zeros(B,T,R, self.hidden_dim).to(inputs.device)
        for t in range(T):
            prev_input = x[:,t,:,:].reshape(B*R,-1)
            for i in range(self.num_rec_layers):
                h_prev_fwd[i], c_prev_fwd[i] = self.forward_hidden_decoder[i](prev_input, (h_prev_fwd[i], c_prev_fwd[i]))
                h_prev_fwd[i] = self.edge2node(h_prev_fwd[i].reshape(B,R,-1), rel_rec)
                h_prev_fwd[i] = self.mlp3f(h_prev_fwd[i])
                h_prev_fwd[i] = self.node2edge(h_prev_fwd[i], rel_rec, rel_send)
                h_prev_fwd[i] = self.linearf(h_prev_fwd[i]).reshape(B*R,-1)
                if i==0:
                    prev_input = torch.cat([x[:,t,:,:].reshape(B*R,-1), h_prev_fwd[i]], dim=-1)
                else:
                    prev_input = torch.cat([h_prev_fwd[i-1], h_prev_fwd[i]], dim=-1)
            out_fwd[:,t,:,:] = h_prev_fwd[-1].reshape(B,R,-1)
        h_prev_bwd = [torch.zeros((B*R, self.hidden_dim)).to(inputs.device) for _ in range(self.num_rec_layers)]
        c_prev_bwd = [torch.zeros((B*R, self.hidden_dim)).to(inputs.device) for _ in range(self.num_rec_layers)]
        out_bwd = torch.zeros(B,T,R, self.hidden_dim).to(inputs.device)
        for t in reversed(range(T)):
            prev_input = x[:,t,:,:].reshape(B*R,-1)
            for i in range(self.num_rec_layers):
                h_prev_bwd[i], c_prev_bwd[i] = self.backward_hidden_decoder[i](prev_input, (h_prev_bwd[i], c_prev_bwd[i]))
                h_prev_bwd[i] = self.edge2node(h_prev_bwd[i].reshape(B,R,-1), rel_rec)
                h_prev_bwd[i] = self.mlp3b(h_prev_bwd[i])
                h_prev_bwd[i] = self.node2edge(h_prev_bwd[i], rel_rec, rel_send)
                h_prev_bwd[i] = self.linearb(h_prev_bwd[i]).reshape(B*R,-1)
                if i==0:
                    prev_input = torch.cat([x[:,t,:,:].reshape(B*R,-1), h_prev_bwd[i]], dim=-1)
                else:
                    prev_input = torch.cat([h_prev_bwd[i-1], h_prev_bwd[i]], dim=-1)
            out_bwd[:,t,:,:] = h_prev_bwd[-1].reshape(B,R,-1)
        out = torch.cat([out_fwd, out_bwd], dim=-1)
        out = self.edge2node(out.reshape(B*T,R,-1), rel_rec)
        return self.mlp_out(out).reshape(B,T,N,-1).transpose(1,2)


class StateGRNNEncoderSmall(nn.Module):
    """Based on https://github.com/ethanfetaya/NRI (MIT License)."""

    def __init__(
        self, n_in, n_hid, n_states, rnn='gru', do_prob=0.0, factor=False
    ):
        super(StateGRNNEncoderSmall, self).__init__()

        self.n_states = n_states
        self.factor = factor
        self.mlp1 = MLP(n_in, n_hid, n_hid, do_prob)
        self.mlp2 = MLP(n_hid * 2, n_hid, n_hid, do_prob)
        self.mlp3 = MLP(n_hid, n_hid, n_hid, do_prob)
        if self.factor:
            self.mlp4 = MLP(n_hid * 3, n_hid, n_hid, do_prob)
        else:
            self.mlp4 = MLP(n_hid * 2, n_hid, n_hid, do_prob)
        if rnn == 'lstm':
            self.rnn_bi = nn.LSTM(n_hid, n_hid, num_layers=3, batch_first=True, bidirectional=True)
            self.rnn_fo = nn.LSTM(2*n_hid, n_hid, num_layers=1, batch_first=True, bidirectional=False)
        elif rnn == 'gru':
            self.rnn_bi = nn.GRU(n_hid, n_hid, num_layers=3, batch_first=True, bidirectional=True)
            self.rnn_fo = nn.GRU(2*n_hid, n_hid, num_layers=1, batch_first=True, bidirectional=False)
        else:
            raise NotImplementedError(rnn + ' is not implemented as a RNN block.')
        self.fc_out = nn.Linear(n_hid, n_states)
        
        print("Using factor graph GRNN encoder.")

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_normal(m.weight.data)
                m.bias.data.fill_(0.1)

    def edge2node(self, x, rel_rec):
        incoming = torch.matmul(rel_rec.t(), x)
        return incoming / incoming.size(1)

    def node2edge(self, x, rel_rec, rel_send):
        # filter the hidden representation of each timestep to consider only the information of the receiver/sender node
        # rel_rec[num_features * num_atoms, num_atoms] * x[num_samples, num_atoms, num_timesteps]
        # receivers, senders: (num_samples, num_atoms,  num_timesteps*num_dims)
        receivers = torch.matmul(rel_rec, x) 
        senders = torch.matmul(rel_send, x)

        # concatecate filtered reciever/sender hidden representation
        edges = torch.cat([senders, receivers], dim=2)
        return edges

    def forward(self, inputs, rel_rec, rel_send):

        # Input has shape: [num_sims, num_atoms, num_timesteps, num_dims]
        B, N, T, D = inputs.size()
        x = inputs.transpose(1,2).reshape(B*T, N, D)
        # New shape: [num_sims*num_timesteps, num_atoms, num_dims]
        x = self.mlp1(x)  # 2-layer ELU net per node
        x = self.node2edge(x, rel_rec, rel_send)
        x = self.mlp2(x)
        x_skip = x

        if self.factor:
            x = self.edge2node(x, rel_rec)
            x = self.mlp3(x)
            x = self.node2edge(x, rel_rec, rel_send)
            x = torch.cat((x, x_skip), dim=2)  # Skip connection
            x = self.mlp4(x)
        else:
            x = self.mlp3(x)
            x = torch.cat((x, x_skip), dim=2)  # Skip connection
            x = self.mlp4(x)
        R = x.size(1)
        x = x.reshape(B,T, R, -1).transpose(1,2).reshape(B*R,T,-1)
        # New shape: [num_sims*num_edges, num_timesteps, num_hid]
        x, _ = self.rnn_bi(x)
        x, _ = self.rnn_fo(x)
        x = x.reshape(B,R,T,-1).transpose(1,2).reshape(B*T,R,-1)
        x = self.edge2node(x, rel_rec).reshape(B,T,N,-1).transpose(1,2)
        return self.fc_out(x)
