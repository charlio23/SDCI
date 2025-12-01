import argparse
import os
import time
from itertools import permutations

import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
from torch.optim.lr_scheduler import StepLR
import torch.utils.data
from tqdm import tqdm
from sklearn.metrics import f1_score

from model.NRIMPMEncoders import AttENC, RNNENC
from datasets.data_loaders import load_springs_data
from model.StateGRNNEncoder import StateGRNNEncoderSmall
from model.MLP_encoder import MLPEncoder, StateEncoderRegion
from model.MLP_decoder import MLPDecoder
from model.NRIMPMEncoders import RNNENC, AttENC
from model.StateDecoder import StateDecoder
from model.EncoderFixed import EncoderFixed
from model.utils import (gumbel_softmax, create_rel_rec_send, simple,
                        my_softmax, edge_accuracy, edge_accuracy_per_sample , calculate_mse_per_sample)

parser = argparse.ArgumentParser(description='Causal discovery metric tester')

parser.add_argument('--name', default='no_name', type=str)
parser.add_argument('--train_root', default='/dataset/train', type=str)
parser.add_argument('--split', default='train', type=str)
parser.add_argument('--metric', default='reconstruct', type=str)
parser.add_argument('--encoder', default='mlp', type=str, metavar='ENC', help='encoder network (mlp or cnn)')
parser.add_argument('--state-decoder', default='region', type=str, metavar='ENC', help='state encoder network (region or recurent): if hidden_states and decoder==baseline ')
parser.add_argument('--decoder', default='baseline', type=str, metavar='ENC', help='decoder network (baseline or dynamic)')
parser.add_argument('--num-states', default=1, type=int, metavar='N', help='number of states')
parser.add_argument('--num-edge-types', default=2, type=int, metavar='N', help='number of edge types')
parser.add_argument('-b', '--batch-size', default=128, type=int,metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--model_path', required=True, type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument("--prediction_steps", type=int, default=10, metavar="N", help="Num steps to predict before re-using teacher forcing.",)
parser.add_argument('--device', default='cpu', type=str, help='use gpu computation')
parser.add_argument("--ACD", action='store_true', help='use ACD encoder')
parser.add_argument("--stateless", action='store_true', help='use stateless decoder')
parser.add_argument("--one_hot", action='store_true', help='use onehot encoding decoder')
parser.add_argument("--experiment", default='linear', type=str, metavar='EXP', help='experiment to perform (linear or springs)')
parser.add_argument("--suffix", default='_springs5', type=str, metavar='SUF', help='suffix')
parser.add_argument('--decoder-fixed', action='store_true', help='use fixed decoder params')
parser.add_argument('--num-atoms', default=1, type=int, metavar='N', help='number of elements')
parser.add_argument("--hidden_states", action='store_true', help='use hidden states mode')
parser.add_argument("--temperature", default=1, type=float, help='temperature term for state distribution')
parser.add_argument('--hidden-size', default=256, type=int, metavar='N', help='hidden size of the model')
parser.add_argument('--dropout', default=0.0, type=float,metavar='DP', help='dropout rate')


import numpy as np
import scipy.stats


def mean_confidence_interval(data, confidence=0.95):
    """Calculates mean and confidence interval from samples such that they lie within m +/- h 
    with the given confidence.

    Args:
        data (np.array): Sample to calculate the confidence interval.
        confidence (float): Confidence of the interval (betwen 0 and 1).
    """
    n = len(data)
    m, se = np.mean(data), scipy.stats.sem(data)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, h

def get_data_params(loader):
    sample, _, graph = next(iter(loader))
    len_graph = graph.size()
    if args.experiment == 'linear' and len(len_graph) == 3:
        num_states = 1
    if args.experiment == 'springs' and len(len_graph) == 2:
        num_states = 1
    else:
        num_states = len_graph[1]
    return sample.size()[1:] + (num_states,)

def get_edge_types(path):
    return np.load(path)['arr_0'].shape[0] + 1

def get_device():
    return args.device

# Load dataset
args = parser.parse_args()
train_dataset, _ = load_springs_data(args.train_root, None, args.suffix, num_atoms=args.num_atoms, num_states=args.num_states, one_hot=args.one_hot, hidden_states=args.hidden_states, split=args.split)
data_loader = torch.utils.data.DataLoader(train_dataset, shuffle=False, batch_size=args.batch_size)

num_edge_types = args.num_edge_types

data_params = get_data_params(data_loader)
num_atoms, seq_len, num_dims, num_states_real = data_params
state_dim = args.num_states
num_states = args.num_states
if args.hidden_states:
    state_dim = 0

print("=> Data params:")
print("    Seq length:", seq_len)
print("    Num atoms:", num_atoms)
print("    Num dims:", num_dims - state_dim)
print("    Num edge types:", num_edge_types)
print("    Num states:", num_states)
if args.split == 'test':
    seq_len //= 2

device = get_device()
print("=> Using device", device)
if args.encoder == 'rnn':
    encoder = RNNENC(num_dims*seq_len, args.hidden_size, num_edge_types, num_states, do_prob=args.dropout).float().to(device)
elif args.encoder == 'att':
    encoder = AttENC(num_dims*seq_len, args.hidden_size, num_edge_types, num_states, do_prob=args.dropout).float().to(device)
elif args.encoder == 'fixed':
    encoder = EncoderFixed(num_atoms, num_edge_types, num_states).float().to(device)
else:
    encoder = MLPEncoder(num_dims*seq_len, args.hidden_size, num_edge_types, num_states, do_prob=args.dropout).float().to(device)
if args.num_states==1:
    state_encoder = None
elif args.state_decoder == 'region':
    state_encoder = StateEncoderRegion(num_dims, 256, num_states).float().to(device)
else:
    #state_encoder = StateGRNNEncoder(num_dims, 64, num_states).float().to(device)
    state_encoder = StateGRNNEncoderSmall(num_dims, 128, num_states).float().to(device)
    
decoder = MLPDecoder(num_dims, num_states, num_edge_types, 256, 256, 256, True).float().to(device)
state_decoder = StateDecoder(num_dims, 64, num_states, num_atoms, args.state_decoder).float().to(device)
checkpoint = torch.load(args.model_path, map_location=torch.device(device))


encoder.load_state_dict(checkpoint['state_dict_encoder'])
encoder.eval()
if state_encoder is not None:
    state_encoder.load_state_dict(checkpoint['state_dict_state_encoder'])
    state_encoder.eval()
decoder.load_state_dict(checkpoint['state_dict_decoder'])
decoder.eval()

edge_acc_list = torch.empty(0).to(device)
reconstr_mse_list = torch.empty(0).to(device)
pred_mse_list = torch.empty(0).to(device)
state_acc_list = torch.empty(0).to(device)
f1_score_edges = []
f1_score_SG = []
f1_score_states = []
with torch.no_grad():
    es = torch.LongTensor(np.array(list(permutations(range(num_atoms), num_edge_types))).T).to(device)
    rel_rec, rel_send = create_rel_rec_send(num_atoms)
    rel_rec, rel_send = rel_rec.to(device), rel_send.to(device)
    best_state_perm = None
    best_edge_perm = None
    for data in tqdm(data_loader):
        sequence, states, graph = data

        # set input-target
        var = Variable(sequence.float(), requires_grad=False).to(device)
        states = states.to(device).float()
        if args.split == 'test':
            var = var[:,:,:seq_len,:]
            state = states[:,:,:seq_len,:]
            if not args.hidden_states:
                values_target = sequence[:, :, 1:seq_len, :-state_dim].float().to(device)
                predict_target = sequence[:, :, seq_len:, :-state_dim].float().to(device).transpose(1,2)
            else:
                values_target = sequence[:, :, 1:seq_len, :].float().to(device)
                predict_target = sequence[:, :, seq_len:, :].float().to(device).transpose(1,2)
            states_target = state[:,:,:seq_len,:].long()
        else:
            target = sequence[:, :, 1:, :].float().to(device)
            states_target = states[:,:,:,:].long()
            if not args.hidden_states:
                values_target = target[:,:,:,:-state_dim]
                state = states_target.float()
            else:
                values_target = target
        if args.num_states != 1:
            if args.state_decoder=='region':
                state_logits = state_encoder(var[:,:,:,:])
            else:
                state_logits = state_encoder(var[:,:,:,:], rel_rec, rel_send)
            state = simple(state_logits, tau=0.5)
        else:
            state = torch.ones_like(var[:,:,:,0:1])
        # compute edge probabilities
        if args.encoder=='att' or args.encoder=='rnn':
            edge_probs = encoder(var, es)    
        elif args.encoder=='fixed':
            edge_probs = encoder().unsqueeze(0).repeat((var.shape[0],1,1,1))
        else:
            edge_probs = encoder(var, rel_rec, rel_send)
        if args.ACD and not args.stateless:
            edge_probs = edge_probs.unsqueeze(2).repeat((1,1,num_states,1))
        # sample using gumble-softmax 
        edges = gumbel_softmax(edge_probs, tau=0.5, hard=True)
        prob = my_softmax(edge_probs, -1)

        if args.metric == 'reconstruct':
            values_out = decoder(var, state, edges, rel_rec, rel_send, args.prediction_steps)
            reconstr_mse = calculate_mse_per_sample(values_out, values_target).detach()
            reconstr_mse_list = torch.cat([reconstr_mse_list, reconstr_mse])
        elif args.metric == 'predict' and args.split == 'test':
            seq_start = var[:,:,seq_len-1:seq_len, :-num_states].transpose(1,2).contiguous().float()
            state = states.transpose(1, 2).contiguous().float()
            rel_type = edges.unsqueeze(1)
            sizes = [
                seq_start.size(0),
                seq_start.size(1) * (seq_len+1),
                seq_start.size(2),
                seq_start.size(3),
            ]
            preds = torch.zeros(sizes).float().to(device)
            last_pred = seq_start
            for i in range(seq_len+1):
                state_start = state[:, seq_len-1+i:seq_len+i, :, 0]
                last_pred = decoder.single_step_forward(last_pred, state_start, rel_rec, rel_send, rel_type)
                preds[:,i,:,:] = last_pred[:,0,:,:].detach()
            pred_mse = calculate_mse_per_sample(preds, predict_target).detach()
            pred_mse_list = torch.cat([pred_mse_list, pred_mse])
        
        
        graph = graph.transpose(1,2).long().to(device)
        
        if state.shape[1] != states_target.shape[1]:
            states_target = states_target.transpose(1,2)
        
        if num_states_real == num_states or args.ACD:
            if best_state_perm is None or (args.state_decoder=='recurent' and not args.encoder=='fixed'):
                perm_list = torch.tensor(list(permutations(range(num_states))))
                permutation_list = []
                best_state_acc = torch.zeros(num_atoms)
                for i in range(num_atoms):
                    for perm in perm_list:
                        state_acc = edge_accuracy(state[:,i,...,perm], states_target[:,i], binary=False)
                        if state_acc > best_state_acc[i]:
                            best_state_acc[i] = state_acc
                            best_state_perm = perm
                    permutation_list.append(best_state_perm.unsqueeze(0))
                permutation_list = torch.cat(permutation_list).to(rel_send.device).float()
                if args.state_decoder =='recurent':
                    aligned_prob = []
                    for i in range(num_states):
                        senders_states = torch.matmul(rel_send, permutation_list[:,i]).long()
                        aligned_prob.append(prob[:,range(prob.shape[1]),senders_states,None,:])
                    aligned_prob = torch.cat(aligned_prob,dim=2)
                else:
                    aligned_prob = prob.transpose(-1,-2)[...,best_state_perm].transpose(-1,-2)
                best_edge_acc = 0
                perm_list = torch.tensor(list(permutations(range(args.num_edge_types))))
                if args.num_states != graph.shape[2]:
                    aligned_prob = aligned_prob.repeat(1,1,graph.shape[2],1)
                for perm in perm_list:
                    edge_acc = edge_accuracy(aligned_prob[..., perm], graph, binary=False)
                    if edge_acc > best_edge_acc:
                        best_edge_acc = edge_acc
                        best_edge_perm = perm
            if args.state_decoder =='recurent':
                aligned_prob = []
                for i in range(num_states):
                    senders_states = torch.matmul(rel_send, permutation_list[:,i]).long()
                    aligned_prob.append(prob[:,range(prob.shape[1]),senders_states,None,:])
                aligned_prob = torch.cat(aligned_prob, dim=2)
                aligned_states = []
                for i in range(num_atoms):
                    aligned_states.append(state[:,i,None][...,permutation_list[i,:].long()])
                aligned_states = torch.cat(aligned_states, dim=1)
                
                if args.split == 'test':
                    B = state.shape[0]
                    for i in range(B):
                        f1_score_states.append(f1_score(states_target[i].squeeze(-1).cpu().numpy().flatten(), 
                            aligned_states[i].argmax(-1).cpu().numpy().flatten(), average='micro'))

                state_acc = edge_accuracy_per_sample(aligned_states, states_target.squeeze(-1))
                state_acc_list = torch.cat([state_acc_list, state_acc])
            else:
                aligned_prob = prob.transpose(-1,-2)[...,best_state_perm].transpose(-1,-2)
                if args.split == 'test':
                    B = state.shape[0]
                    for i in range(B):
                        f1_score_states.append(f1_score(states_target[i].squeeze(-1).cpu().numpy().flatten(), 
                            state[i][..., best_state_perm].argmax(-1).cpu().numpy().flatten(), average='micro'))

                state_acc = edge_accuracy_per_sample(state[..., best_state_perm], states_target.squeeze(-1))
                
                state_acc_list = torch.cat([state_acc_list, state_acc])

            if args.num_states != graph.shape[2]:
                aligned_prob = aligned_prob.repeat(1,1,graph.shape[2],1)
            if args.split == 'test':
                B = state.shape[0]

                for i in range(B):
                    f1_score_SG.append(f1_score((graph[i].sum(-1) > 0).cpu().numpy().flatten(), 
                        (aligned_prob[i].argmax(-1).sum(-1) > 0).cpu().numpy().flatten(), average='micro'))
                    f1_score_edges.append(f1_score(graph[i].cpu().numpy().flatten(), 
                        aligned_prob[i][..., best_edge_perm].argmax(-1).cpu().numpy().flatten(), average='micro'))
            
            edge_acc = edge_accuracy_per_sample(aligned_prob[..., best_edge_perm], graph)
            edge_acc_list = torch.cat([edge_acc_list, edge_acc])
        else:
            B = graph.shape[0]
            for i in range(B):
                f1_score_SG.append(f1_score((graph[i].sum(-1) > 0).cpu().numpy().flatten(), 
                    (prob[i].argmax(-1).sum(-1) > 0).cpu().numpy().flatten(), average='micro'))

    print(edge_acc_list)
    edge_acc_mean, edge_acc_h = mean_confidence_interval(edge_acc_list.cpu().numpy())
    print("=> Edge accuracy is", edge_acc_mean*100, "+/-", edge_acc_h*100)
    if args.hidden_states:
        state_acc_mean, state_acc_h = mean_confidence_interval(state_acc_list.cpu().numpy())
        print("=> State. Decoder Accuracy is", state_acc_mean*100, "+/-", state_acc_h*100)
    if args.metric == 'reconstruct':
        reconstr_mse_mean, reconstr_mse_h = mean_confidence_interval(reconstr_mse_list.cpu().numpy())
        print("=> Reconstr. MSE is", reconstr_mse_mean, "+/-", reconstr_mse_h)
        np.save('results/metrics_' + args.name + 'reconstr_mse', reconstr_mse_list.detach().cpu().numpy())

    elif args.metric =='predict':
        pred_mse_mean, pred_mse_h = mean_confidence_interval(pred_mse_list.cpu().numpy())
        print("=> Prediction MSE is", pred_mse_mean, "+/-", pred_mse_h)

    os.makedirs('results/', exist_ok=True)
    np.save('results/metrics_' + args.name + 'edge_acc', edge_acc_list.detach().cpu().numpy())
    np.save('results/metrics_' + args.name + 'state_acc', state_acc_list.detach().cpu().numpy())

    if args.split == 'test':
        f1_score_states = np.array(f1_score_states)
        f1_score_edges = np.array(f1_score_edges)
        f1_score_SG = np.array(f1_score_SG)
        edge_f1_mean, edge_f1_h = mean_confidence_interval(f1_score_edges)
        sg_f1_mean, sg_f1_h = mean_confidence_interval(f1_score_SG)
        print("=> Summary Graph F1 Score is", sg_f1_mean*100, "+/-", sg_f1_h*100)
        print("=> Edge F1 Score is", edge_f1_mean*100, "+/-", edge_f1_h*100)
        state_f1_mean, state_f1_h = mean_confidence_interval(f1_score_states)
        print("=> State. Decoder F1 Score is", state_f1_mean*100, "+/-", state_f1_h*100)
        np.save('results/metrics_' + args.name + '_edge_f1_score', f1_score_edges)
        np.save('results/metrics_' + args.name + '_sg_f1_score', f1_score_SG)
        np.save('results/metrics_' + args.name + '_state_f1_score', f1_score_states)
