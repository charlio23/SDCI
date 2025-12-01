import torch
from torch import nn
from torch.nn import functional as F
from model.modules_ACD import MLP
from model.utils import gumbel_softmax, simple

class MLPDecoder(nn.Module):
    """Based on https://github.com/ethanfetaya/NRI (MIT License)."""

    def __init__(
        self,
        n_in_node,
        num_states,
        edge_types,
        msg_hid,
        msg_out,
        n_hid,
        hidden_states=False,
        do_prob=0.0,
        skip_first=True,
        ball_cond=False,
        embedding=False,
        num_atoms=None
    ):
        super(MLPDecoder, self).__init__()
        self.msg_fc1 = nn.ModuleList(
            [nn.Linear(2 * n_in_node + (16 if embedding else 0), msg_hid) for _ in range(edge_types)]
        )
        self.msg_fc2 = nn.ModuleList(
            [nn.Linear(msg_hid, msg_out) for _ in range(edge_types)]
        )
        self.msg_out_shape = msg_out
        self.skip_first_edge_type = skip_first
        
        n_in = n_in_node + msg_out
        if ball_cond:
            n_in += 6
        self.embedding = embedding
        if embedding:
            self.node_embedding = nn.Embedding(num_atoms, 16)
            n_in += 16
        self.out_fc1 = nn.Linear(n_in, n_hid)
        self.out_fc2 = nn.Linear(n_hid, n_hid)
        self.out_fc3 = nn.Linear(n_hid, n_in_node)
        
        print("Using learned interaction net decoder.")
        self.num_states = num_states
        self.hidden_states = hidden_states
        self.dropout_prob = do_prob

    def single_step_forward(
        self, single_timestep_inputs, states, rel_rec, rel_send, single_timestep_rel_type, ball_pos=None
        ):

        # single_timestep_inputs has shape
        # [batch_size, num_timesteps, num_atoms, num_dims]

        # single_timestep_rel_type has shape:
        # [batch_size, num_timesteps, num_atoms*(num_atoms-1), num_edge_types]
        # Node2edge
        B,T,num_atoms,num_dims = single_timestep_inputs.shape
        receivers = torch.matmul(rel_rec, single_timestep_inputs)
        if self.embedding:
            sender_features = torch.cat([single_timestep_inputs,self.node_embedding.weight[None,None].expand(B,T,-1,-1)],dim=-1)
        else:
            sender_features = single_timestep_inputs
        senders = torch.matmul(rel_send, sender_features)
        if not self.hidden_states:
            senders_states = torch.matmul(rel_send, states.unsqueeze(-1)).long()
        else:
            senders_states = torch.matmul(rel_send, states)
        pre_msg = torch.cat([senders, receivers], dim=-1)
        all_msgs = torch.zeros(
            pre_msg.size(0), pre_msg.size(1), pre_msg.size(2), self.msg_out_shape
        )
        if ball_pos is not None:
            skip_connection = torch.cat([single_timestep_inputs[:,:,:,:], ball_pos.expand(-1,-1,num_atoms,-1)], dim=-1)
        else:
            skip_connection = sender_features

        if single_timestep_inputs.is_cuda:
            all_msgs = all_msgs.to(single_timestep_inputs.device)

        if self.skip_first_edge_type:
            start_idx = 1
        else:
            start_idx = 0

        # Run separate MLP for every edge type
        # NOTE: To exclude one edge type, simply offset range by 1
        batch, num_timesteps, num_rel, num_states, _ = single_timestep_rel_type.size()
        for i in range(start_idx, len(self.msg_fc2)):
            msg = F.leaky_relu(self.msg_fc1[i](pre_msg))
            msg = F.dropout(msg, p=self.dropout_prob)
            msg = F.leaky_relu(self.msg_fc2[i](msg))

            rel_types = single_timestep_rel_type[:,:,:,:,i].reshape(batch*num_timesteps*num_rel, num_states)
            if not self.hidden_states:
                rel_types = rel_types[range(batch*num_timesteps*num_rel), senders_states.reshape(-1)].reshape(batch,num_timesteps,num_rel,1)
            else:
                rel_types = torch.sum(rel_types*senders_states.reshape(batch*num_timesteps*num_rel, num_states),dim=-1).reshape(batch,num_timesteps,num_rel,1)
            msg = msg * rel_types

            all_msgs += msg

        # Aggregate all msgs to receiver
        agg_msgs = all_msgs.transpose(-2, -1).matmul(rel_rec).transpose(-2, -1)
        agg_msgs = agg_msgs.contiguous()

        # Skip connection
        aug_inputs = torch.cat([skip_connection, agg_msgs], dim=-1)
        # Output MLP
        pred = F.dropout(F.leaky_relu(self.out_fc1(aug_inputs)), p=self.dropout_prob)
        pred = F.dropout(F.leaky_relu(self.out_fc2(pred)), p=self.dropout_prob)
        pred = self.out_fc3(pred)

        next_pos = single_timestep_inputs + pred
        #next_state_distrib = pred[:,:,:,num_dims-1:]
        # Predict position/velocity difference
        return next_pos

    def forward(self, inputs, state, rel_type, rel_rec, rel_send, pred_steps=1, ball_pos=None):
        # NOTE: Assumes that we have the same graph across all samples.

        inputs = inputs.transpose(1, 2).contiguous()
        state = state.transpose(1, 2).contiguous()
        time_steps = inputs.size(1)
        assert pred_steps <= time_steps
        preds = []
        # Only take n-th timesteps as starting points (n: pred_steps)
        if not self.hidden_states:
            last_pred = inputs[:, 0::pred_steps, :, :-self.num_states]
        else:
            last_pred = inputs[:, 0::pred_steps, :, :]
        sizes = [
            rel_type.size(0),
            last_pred.size(1),
            rel_type.size(1),
            rel_type.size(2),
            rel_type.size(3),
        ]  # batch, sequence length, interactions between particles, interaction types
        rel_type = rel_type.unsqueeze(1).expand(
            sizes
        )
        # Run n prediction steps
        for step in range(0, pred_steps):
            ball_in = ball_pos[:, step::pred_steps, :, :] if ball_pos is not None else None
            if not self.hidden_states:
                states = state[:, step::pred_steps, :, 0]
            else:
                states = state[:, step::pred_steps, :, :]
            if (states.size(1) != last_pred.size(1)):
                if not self.hidden_states:
                    states = state[:, step-1::pred_steps, :, 0]
                else:
                    states = state[:, step-1::pred_steps, :, :]
            last_pred = self.single_step_forward(
                last_pred, states, rel_rec, rel_send, rel_type, ball_in
            )
            #state = torch.argmax(last_state, dim=-1).unsqueeze(-1).float()
            # store logits for states
            #last_pred_distrib = torch.cat([last_result, last_state], dim=-1)
            # store state classification results
            #last_pred = torch.cat([last_result, state], dim=-1)
            preds.append(last_pred)

        sizes = [
            preds[0].size(0),
            preds[0].size(1) * pred_steps,
            preds[0].size(2),
            preds[0].size(3),
        ]

        output = torch.zeros(sizes).to(inputs.device)

        # Re-assemble correct timeline
        for i in range(len(preds)):
            output[:, i::pred_steps, :, :] = preds[i]
        pred_all = output[:, : (inputs.size(1) - 1), :, :]

        return pred_all.transpose(1, 2).contiguous()

    def predict_sequence(self, seq_start, state_function, rel_type, rel_rec, rel_send, pred_steps=100, ball_pos=None, state_type='region', state_info=None):
        last_pred = seq_start.transpose(1, 2).contiguous()
        state_info = state_info.transpose(1, 2).contiguous()
        
        sizes = [
            rel_type.size(0),
            last_pred.size(1)*pred_steps,
            last_pred.size(2),
            last_pred.size(3),
        ]  # batch, sequence length, interactions between particles, interaction types
        rel_type = rel_type.unsqueeze(1)
        preds = torch.zeros(sizes).float().to(seq_start)
        for i in range(pred_steps):
            ball_in = ball_pos[:,i:i+1] if ball_pos is not None else None
            if i==0 or state_info.shape[1] > 1:
                last_state = state_info[:,i:i+1]
            last_result = self.single_step_forward(last_pred, last_state, rel_rec, rel_send, rel_type, ball_in)
            preds[:,i,:,:] = last_result[:,0,:,:].detach()
            if state_function is not None:
                if state_type=='region':
                    # input: x_t
                    state_in = last_result
                else:
                    # input: x_t-1, x_t, s_t-1
                    state_in = torch.cat([last_pred, last_result, last_state], dim=-1)
                state_logits = state_function(state_in)
                last_state = simple(state_logits, tau=0.5)
            last_pred = last_result
        return preds