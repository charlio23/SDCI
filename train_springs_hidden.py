import argparse
import os
import time

import numpy as np
from itertools import permutations
from matplotlib import pyplot as plt
from tensorboardX import SummaryWriter
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim
from torch.optim.lr_scheduler import StepLR
import torch.utils.data
from itertools import permutations

from model.NRIMPMEncoders import AttENC, RNNENC
from datasets.data_loaders import load_springs_data
from model.StateGRNNEncoder import StateGRNNEncoderSmall
from model.MLP_encoder import MLPEncoder, StateEncoderRegion
from model.MLP_decoder import MLPDecoder
from model.StateDecoder import StateDecoder
from model.EncoderFixed import EncoderFixed
from model.utils import (gumbel_softmax, create_rel_rec_send, nll_gaussian, simple,
                        my_softmax, kl_categorical, edge_accuracy, calculate_mse)


parser = argparse.ArgumentParser(description='Causal discovery trainer')

parser.add_argument('--name', required=True, type=str, help='Name of the experiment')
parser.add_argument('--train_root', default='/dataset/train', type=str)
parser.add_argument('--train_idx', default='', type=str)
parser.add_argument('--epochs', default=500, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--step_size', default=200, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--encoder', default='mlp', type=str, metavar='ENC', help='encoder network (mlp or cnn)')
parser.add_argument('--hidden-size', default=256, type=int, metavar='N', help='hidden size of the model')
parser.add_argument('--state-decoder', default='region', type=str, metavar='ENC', help='state decoder network (region or recurrent)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('--num-states', default=1, type=int, metavar='N', help='number of states')
parser.add_argument('--num-edge-types', default=2, type=int, metavar='N', help='number of edge-types')
parser.add_argument('--num-atoms', default=1, type=int, metavar='N', help='number of elements')
parser.add_argument('-b', '--batch-size', default=128, type=int,metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr-encoder', default=5e-4, type=float,metavar='LR', help='initial learning rate for the encoder')
parser.add_argument('--lr-decoder', default=5e-4, type=float,metavar='LR', help='initial learning rate for the decoder')
parser.add_argument('--dropout', default=0.0, type=float,metavar='DP', help='Dropout rate')
parser.add_argument("--prediction_steps", type=int, default=10, metavar="N", help="Num steps to predict before re-using teacher forcing.",)
parser.add_argument('--print-freq', '-p', default=10, type=int, metavar='N', help='print frequency (default: 10)')
parser.add_argument('--load', required=False, type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('--device', default='cpu', type=str, help='device for gpu computation')
parser.add_argument("--suffix", default='_springs5', type=str, metavar='SUF', help='suffix')
parser.add_argument("--logdir", default='./runs', type=str, metavar='RUN', help='Tensorboard log directory')
parser.add_argument("--init_temperature", default=1, type=float, help='temperature term for state distribution')
parser.add_argument("--sampler", default='gumble', type=str, metavar='SAMP', help='Sampling method')
parser.add_argument("--data-size", default=-1, type=int, help='Dataset size (-1) for all available data')
parser.add_argument('--sparsity', default=0.5, type=float,metavar='SP', help='sparsity')
parser.add_argument('--kl_mult', default=1, type=float,metavar='KL', help='KL Multiplier')


def main():
    global args, writer
    args = parser.parse_args()
    os.makedirs(args.logdir, exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join(args.logdir, args.name))
    print(args)
    torch.autograd.set_detect_anomaly(True)
    # Init data
    train_root = args.train_root
    train_dataset, _ = load_springs_data(train_root, None if args.train_idx=='' else args.train_idx, 
                                        args.suffix, num_atoms=args.num_atoms, num_states=args.num_states, 
                                        one_hot=False, hidden_states=True, data_size = args.data_size)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    print("=> Loaded train data, length = ", len(train_dataset))
    num_edge_types = args.num_edge_types
    data_params = get_data_params(train_loader)
    num_atoms, seq_len, num_dims, num_states = data_params
    num_states = args.num_states
    print("=> Data params:")
    print("    Seq length:", seq_len)
    print("    Num atoms:", num_atoms)
    print("    Num dims:", num_dims)
    print("    Num edge types:", num_edge_types)
    print("    Num states:", num_states)

    # Create model
    device = args.device
    print("=> Using device", device)
    print("=> Initializing models")
    if args.encoder =='fixed':
        encoder = EncoderFixed(num_atoms, num_edge_types, num_states).float().to(device)
    elif args.encoder == 'rnn':
        encoder = RNNENC(num_dims*seq_len, args.hidden_size, num_edge_types, num_states, do_prob=args.dropout).float().to(device)
    elif args.encoder == 'att':
        encoder = AttENC(num_dims*seq_len, args.hidden_size, num_edge_types, num_states, do_prob=args.dropout).float().to(device)
    else:
        encoder = MLPEncoder(num_dims*seq_len, args.hidden_size, num_edge_types, num_states).float().to(device)
    if args.num_states==1:
        state_encoder = None
        state_decoder = None
    elif args.state_decoder == 'region':
        state_encoder = StateEncoderRegion(num_dims, 256, num_states).float().to(device)
    else:
        #state_encoder = StateGRNNEncoder(num_dims, 64, num_states).float().to(device)
        state_encoder = StateGRNNEncoderSmall(num_dims, 128, num_states).float().to(device)
    if args.num_states != 1:
        state_decoder = StateDecoder(num_dims, 64, num_states, num_atoms, args.state_decoder).float().to(device)
    decoder = MLPDecoder(num_dims, num_states, num_edge_types, 256, 256, 256, True).float().to(device)

    
    # Optionally resume from a checkpoint
    if args.load: 
        for (path, enc, dec, st_enc) in [(args.load, encoder, decoder, state_encoder)]:
            if os.path.isfile(path):
                print("=> loading checkpoint '{}'".format(args.load))
                checkpoint = torch.load(path)
                args.start_epoch = checkpoint['epoch']
                enc.load_state_dict(checkpoint['state_dict_encoder'])
                dec.load_state_dict(checkpoint['state_dict_decoder'])
                if st_enc is not None:
                    st_enc.load_state_dict(checkpoint['state_dict_state_encoder'])
                print("=> loaded checkpoint '{}' (epoch {})"
                      .format(path, checkpoint['epoch']))
            else:
                print("=> no checkpoint found at '{}'".format(path))

    print(encoder)
    print(state_encoder)
    print(decoder)

    # Define optimizer
    optim_params = [{
                'params':   encoder.parameters(),
                'lr':       args.lr_encoder
            }]
    
    if state_encoder is not None:
        optim_params += [{
            'params':   state_encoder.parameters(),
            'lr':       args.lr_decoder
        }]
        optim_params += [{
                'params':   state_decoder.parameters(),
                'lr':       args.lr_decoder
            }]
    optim_params += [{
            'params':   decoder.parameters(),
            'lr':       args.lr_decoder
        }]

    optimizer = torch.optim.Adam(optim_params)
    gamma = 0.5
    scheduler = StepLR(optimizer, step_size=args.step_size, gamma=gamma)
    init_temperature = args.init_temperature
    gamma_temp = 0.75
    final_temperature = 0.5
    temperature_state = init_temperature
    # Training loop
    print("-------- Training started --------")
    calculate_metrics(train_loader, num_atoms, num_edge_types, num_states, seq_len, encoder, state_encoder, decoder, state_decoder, temperature_state, 0)
    for epoch in range(args.start_epoch, args.epochs):
        print("=> Epoch", epoch+1, "started.")
        # train for one epoch
        train_one_epoch(train_loader, num_atoms, num_edge_types, num_states, seq_len, encoder, state_encoder, decoder, state_decoder, optimizer, temperature_state, epoch)
        scheduler.step()
        # save checkpoint
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict_encoder': encoder.state_dict(),
            'state_dict_decoder': decoder.state_dict(),
            'state_dict_state_encoder': state_encoder.state_dict() if state_encoder is not None else None,
            'state_dict_state_decoder': state_decoder.state_dict() if state_decoder is not None else None
        }, filename=args.name)
        print("=> Epoch", epoch+1, "finished.")
        # calculate metrics and log to console
        calculate_metrics(train_loader, num_atoms, num_edge_types, num_states, seq_len, encoder, state_encoder, decoder, state_decoder, temperature_state, epoch+1)
        if epoch > 10:
            temperature_state = max(final_temperature, temperature_state*gamma_temp)
        print("-------- End of epoch --------", temperature_state)


def train_one_epoch(train_loader, num_atoms, num_edge_types, num_states, seq_len, encoder, state_encoder, decoder, state_decoder, optimizer, temperature_state, epoch):
    device = args.device
    encoder.train()
    if state_encoder is not None:
        state_encoder.train()
    decoder.train()
    es = torch.LongTensor(np.array(list(permutations(range(num_atoms), 2))).T).to(device)
    rel_rec, rel_send = create_rel_rec_send(num_atoms)
    rel_rec, rel_send = rel_rec.to(device), rel_send.to(device)

    end = time.time()
    for i, data in enumerate(train_loader):
        sequence, state, graph = data
        target = sequence[:, :, 1:, :].float().to(device)
        # set input-target
        var = Variable(sequence.float(), requires_grad=True).to(device)
        optimizer.zero_grad()
        # compute state probabilities
        if args.num_states != 1:
            if args.state_decoder=='region':
                state_logits = state_encoder(var[:,:,:,:])
            else:
                state_logits = state_encoder(var[:,:,:,:], rel_rec, rel_send)
            state_samp = simple(state_logits, tau=temperature_state)
            #state_samp = gumbel_softmax(state_logits, tau=temperature_state, hard=False)
            state_prob = my_softmax(state_logits, -1)
            if args.state_decoder=='region':
                state_in = var
            else:
                state_in = torch.cat([var[:,:,:-1,:], var[:,:,1:,:], state_samp[:,:,:-1,:]], dim=-1)
            state_prior = state_decoder(state_in)
        else:
            state_samp = torch.ones_like(var[:,:,:,0:1])
            state_prob = None
        # compute edge probabilities
        if args.encoder =='fixed':
            edge_logits = encoder().unsqueeze(0).repeat(var.size(0),1,1,1)
        elif args.encoder=='att' or args.encoder=='rnn':
            edge_logits = encoder(var, es)    
        else:
            edge_logits = encoder(var, rel_rec, rel_send)
        # sample using gumble-softmax 
        if args.sampler=='gumble':
            edges = gumbel_softmax(edge_logits, tau=0.5, hard=False)
        else:
            edges = simple(edge_logits, tau=0.5)
        edge_prob = my_softmax(edge_logits, -1)

        output = decoder(var, state_samp, edges, rel_rec, rel_send, args.prediction_steps)
        # gather values and states
        values_out = output[:,:,:,:]
        # record loss
        nll = nll_gaussian(values_out, target, 5e-5)
        if args.state_decoder=='region' or args.num_states==1:
            kl = 0
        else:
            kl = kl_categorical(state_prob[:,:,1:], torch.log(state_prior + 1e-16), num_atoms=1)
        for k in range(num_states):
            if args.num_edge_types==3:
                kl += args.kl_mult*kl_categorical(edge_prob[:,:,k,:], torch.log(torch.tensor([0.5, 0.25, 0.25]).to(device)), num_atoms=1)
            else:
                kl += args.kl_mult*kl_categorical(edge_prob[:,:,k,:], torch.log(torch.tensor([args.sparsity, 1-args.sparsity]).to(device)), num_atoms=1)

            
        loss = nll + kl
        # compute gradient and do SGD step
        loss.backward()
        if state_encoder is not None:
            torch.nn.utils.clip_grad_norm_(state_encoder.parameters(), 5.0)
        torch.nn.utils.clip_grad_norm_(encoder.parameters(), 5.0)
        torch.nn.utils.clip_grad_norm_(decoder.parameters(), 5.0)
        optimizer.step()

        # measure elapsed time
        batch_time = time.time() - end
        mse = F.mse_loss(values_out, target, reduction='none').sum()/args.batch_size


        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time:.3f}\t'
                  'Loss {loss:.4e}\t MSE: {mse:.4e}\t'.format(
                   epoch, i, len(train_loader), batch_time=batch_time, loss=loss, mse=mse))
        writer.add_scalar('data/nll_loss', nll, i + epoch*len(train_loader))
        writer.add_scalar('data/kl_loss', kl, i + epoch*len(train_loader))
        writer.add_scalar('data/total_loss', loss, i + epoch*len(train_loader))
        writer.add_scalar('data/mse_loss', mse, i + epoch*len(train_loader))

def save_checkpoint(state, filename='model'):
    os.makedirs("models_saved/", exist_ok=True)
    torch.save(state, "models_saved/" + filename + '_latest.pth.tar')

def print_net_params(net):
    params_msg = []
    for parameter in net.parameters():
        msg = np.array2string(parameter.detach().cpu().numpy())
        print(msg)
        params_msg.append(msg)
    return str(params_msg[1:2])
    
def print_gt_matrices(path):
    msg = np.array2string(np.load(path)['arr_0'])
    print(msg)
    return msg

def calculate_metrics(train_loader, num_atoms, num_edge_types, num_states, seq_len, encoder, state_encoder, decoder, state_decoder, temperature_state, epoch):
    # Calculate metrics from a single batch
    device = args.device
    sequence, state, graph = next(iter(train_loader))
    graph = graph.transpose(1,2).unsqueeze(-1).long()
    rel_rec, rel_send = create_rel_rec_send(num_atoms)
    rel_rec, rel_send = rel_rec.to(device), rel_send.to(device)
    es = torch.LongTensor(np.array(list(permutations(range(num_atoms), 2))).T).to(device)
    # set input-target
    var = Variable(sequence.float(), requires_grad=False).to(device)
    state = state.to(device).float()
    target = sequence[:, :, 1:, :].float().to(device)
    states_target = state[:,:,:,:].long() #remove transpose on necessity
    values_target = target
    #target_graph = adj_mat_to_target(graph.to(device), rel_rec, rel_send).long()
    # compute edge probabilities
    enc_input = var
    if args.num_states != 1:
        if args.state_decoder=='region':
            state_logits = state_encoder(var[:,:,:,:])
        else:
            state_logits = state_encoder(var[:,:,:,:], rel_rec, rel_send)
        state_samp = simple(state_logits, tau=0.5)
    else:
        state_samp = torch.ones_like(var[:,:,:,0:1])
    if args.encoder =='fixed':
        edge_probs = encoder().unsqueeze(0).repeat(var.size(0),1,1,1)
    elif args.encoder=='att' or args.encoder=='rnn':
        edge_probs = encoder(enc_input, es)
    else:
        edge_probs = encoder(enc_input, rel_rec, rel_send)
    # sample using gumble-softmax 
    edges = simple(edge_probs, tau=0.5)
    prob = my_softmax(edge_probs, -1)
    # compute decoder predictions
    output = decoder(var, state_samp, edges, rel_rec, rel_send, args.prediction_steps)
    values_out = output[:,:,:,:]

    # encoder edge classification accuracy
    graph = graph.to(device)

    perm_list = torch.tensor(list(permutations(range(num_states))))
    best_state_perm = None
    permutation_list = []
    best_state_acc = torch.zeros(num_atoms)
    for i in range(num_atoms):
        for perm in perm_list:
            state_acc = edge_accuracy(state_samp[:,i,...,perm], states_target[:,i], binary=False)
            if state_acc > best_state_acc[i]:
                best_state_acc[i] = state_acc
                best_state_perm = perm
        permutation_list.append(best_state_perm.unsqueeze(0))
    permutation_list = torch.cat(permutation_list).to(rel_send.device).float()
    writer.add_scalar('metrics/decoder_state_accuracy', best_state_acc.mean(), epoch)
    if args.state_decoder=='region':
        aligned_prob = prob.transpose(-1,-2)[...,best_state_perm].transpose(-1,-2)
    else:
        aligned_prob = []
        for i in range(num_states):
            senders_states = torch.matmul(rel_send, permutation_list[:,i]).long()
            aligned_prob.append(prob[:,range(prob.shape[1]),senders_states,None,:])
        aligned_prob = torch.cat(aligned_prob,dim=2)
    best_edge_acc = 0
    perm_list = torch.tensor(list(permutations(range(args.num_edge_types))))
    #perm_list = torch.hstack((torch.zeros(perm_list.shape[0],1), perm_list)).long()
    if args.num_states != graph.shape[2]:
        aligned_prob = aligned_prob.repeat(1,1,graph.shape[2],1)
    for perm in perm_list:
        edge_acc = edge_accuracy(aligned_prob[..., perm], graph, binary=False)
        if edge_acc > best_edge_acc:
            best_edge_acc = edge_acc
            best_edge_perm = perm 
    writer.add_scalar('metrics/encoder_edge_accuracy', best_edge_acc, epoch)
    # decoder state classification accuracy
    # MSE positions
    mse_positions = calculate_mse(values_out, values_target)
    writer.add_scalar('metrics/mse_positions', mse_positions, epoch)

    fig = plt.figure(figsize=(20,10))
    axes = plt.gca()
    plt.axis('on')
    for i in range(args.num_atoms):
        plt.subplot(args.num_atoms,1,i+1)
        plt.plot(range(seq_len), sequence[0,i,:,0].detach().cpu().numpy())
        plt.plot(range(seq_len), sequence[0,i,:,1].detach().cpu().numpy())
        plt.plot(range(seq_len), state_samp[0,i,:][...,best_state_perm].argmax(-1).detach().cpu().numpy(), linewidth=2, c='black')
        plt.plot(range(seq_len), states_target[0,i,:].detach().cpu().numpy(), linewidth=2, c='green')
        plt.plot(range(seq_len), [0.5]*seq_len, linewidth=2, c='red')
    writer.add_figure('metrics/state_viz', fig, epoch)

     


def get_edge_types(path):
    return np.load(path)['arr_0'].shape[0] + 1

def get_data_params(loader):
    sample, _, graph = next(iter(loader))
    print(sample.size())
    len_graph = graph.size()
    if len(len_graph) == 2:
        num_states = 1
    else:
        num_states = len_graph[1]
    return sample.size()[1:] + (num_states,)

if __name__ == '__main__':
    main()
