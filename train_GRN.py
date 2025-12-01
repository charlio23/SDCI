import argparse
import os
import time
import sys

import numpy as np
from tensorboardX import SummaryWriter
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import torch.optim
from torch.optim.lr_scheduler import StepLR
import torch.utils.data
from tqdm import tqdm
from itertools import permutations
from sklearn.metrics import roc_auc_score

from datasets.data_loaders import GRN
from model.StateGRNNEncoder import StateGRNNEncoderSmall
from model.MLP_encoder import StateEncoderRegion
from model.MLP_decoder import MLPDecoder
from model.StateDecoder import StateDecoder
from model.EncoderFixed import EncoderFixed
from model.utils import (create_rel_rec_send, nll_gaussian, simple, gumbel_softmax,
                        my_softmax, kl_categorical, calculate_mse)


parser = argparse.ArgumentParser(description='Causal discovery trainer')

parser.add_argument('--name', required=True, type=str, help='Name of the experiment')
parser.add_argument('--train_root', default='/dataset/train', type=str)
parser.add_argument('--epochs', default=500, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--step_size', default=200, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--encoder', default='mlp', type=str, metavar='ENC', help='encoder network (mlp or cnn)')
parser.add_argument('--hidden-dim', default=256, type=int, metavar='N', help='hidden size of the model')
parser.add_argument('--state-decoder', default='region', type=str, metavar='ENC', help='state decoder network (region or recurrent)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('--num-states', default=1, type=int, metavar='N', help='number of states')
parser.add_argument('--num-atoms', default=52, type=int, metavar='N', help='number of elements')
parser.add_argument('--num-dims', default=1, type=int, metavar='N', help='number of dimensions (lags)')
parser.add_argument('--num-edge-types', default=2, type=int, metavar='N', help='number of edge-types')
parser.add_argument('-b', '--batch-size', default=128, type=int,metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr-encoder', default=5e-4, type=float,metavar='LR', help='initial learning rate for the encoder')
parser.add_argument('--lr-decoder', default=5e-4, type=float,metavar='LR', help='initial learning rate for the decoder')
parser.add_argument("--prediction_steps", type=int, default=10, metavar="N", help="Num steps to predict before re-using teacher forcing.")
parser.add_argument('--print-freq', '-p', default=10, type=int, metavar='N', help='print frequency (default: 10)')
parser.add_argument('--load', required=False, type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('--device', default='cpu', type=str, help='device for gpu computation')
parser.add_argument("--init_temperature", default=1, type=float, help='temperature term for state distribution')
parser.add_argument('--sparsity', default=0.5, type=float,metavar='SP', help='sparsity')
parser.add_argument('--kl_mult', default=1, type=float,metavar='KL', help='KL Multiplier')
parser.add_argument("--hidden_states", action='store_false', help='use hidden states mode')
parser.add_argument("--masked", action='store_true', help='use masked data')
parser.add_argument("--embedding", action='store_true', help='use embedding')

def main():
    global args, writer
    args = parser.parse_args()
    os.makedirs("./runs_GRN", exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join("./runs_GRN", args.name))
    print(args)
    torch.autograd.set_detect_anomaly(True)
    # Init data
    train_dataset = GRN(args.train_root, args.num_atoms, args.num_dims, args.hidden_states, args.masked)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    test_dataset = GRN(args.train_root, args.num_atoms, args.num_dims, args.hidden_states, args.masked)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size*2, shuffle=True, num_workers=4)
    print("=> Loaded train data, length = ", len(train_dataset))
    num_edge_types = args.num_edge_types
    num_states = args.num_states if args.hidden_states else 2
    seq_len = 50
    num_atoms = args.num_atoms 
    num_dims = args.num_dims
    if args.encoder=='acd':
        num_states = 1    

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

    encoder = EncoderFixed(num_atoms, num_edge_types, num_states).float().to(device)
    state_encoder = None
    state_decoder = None
    if args.hidden_states:
        if args.state_decoder == 'region':
            state_encoder = StateEncoderRegion(num_dims, args.hidden_dim, num_states).float().to(device)
        else:
            #state_encoder = StateGRNNEncoder(num_dims, 64, num_states).float().to(device)
            state_encoder = StateGRNNEncoderSmall(num_dims, args.hidden_dim, num_states).float().to(device)
        state_decoder = StateDecoder(num_dims, args.hidden_dim, num_states, num_atoms, args.state_decoder).float().to(device)    
    decoder = MLPDecoder(num_dims, num_states, num_edge_types, args.hidden_dim, args.hidden_dim, args.hidden_dim, True,
                         ball_cond=False, embedding=args.embedding, num_atoms=args.num_atoms).float().to(device)
    
    # Optionally resume from a checkpoint
    if args.load: 
        for (path, enc, dec, st_enc, st_dec) in [(args.load, encoder, decoder, state_encoder, state_decoder)]:
            if os.path.isfile(path):
                print("=> loading checkpoint '{}'".format(args.load))
                checkpoint = torch.load(path)
                args.start_epoch = checkpoint['epoch']
                enc.load_state_dict(checkpoint['state_dict_encoder'])
                dec.load_state_dict(checkpoint['state_dict_decoder'])
                if st_enc is not None:
                    st_enc.load_state_dict(checkpoint['state_dict_state_encoder'])
                    st_dec.load_state_dict(checkpoint['state_dict_state_decoder'])
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
            'params':   decoder.parameters(),
            'lr':       args.lr_decoder
        }]
    if state_decoder is not None:
        optim_params += [{
                'params':   state_decoder.parameters(),
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
    calculate_metrics(test_loader, num_atoms, num_edge_types, num_states, seq_len, encoder, state_encoder, decoder, state_decoder, temperature_state, 0)
    for epoch in range(args.start_epoch, args.epochs):
        print("=> Epoch", epoch+1, "started.")
        # train for one epoch
        train_one_epoch(train_loader, num_atoms, num_edge_types, num_states, seq_len, encoder, state_encoder, decoder, state_decoder, optimizer, temperature_state, epoch)
        scheduler.step()
        print("=> Epoch", epoch+1, "finished.")
        # calculate metrics and log to console
        if (epoch+1)%10==0:
            # save checkpoint
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict_encoder': encoder.state_dict(),
                'state_dict_decoder': decoder.state_dict(),
                'state_dict_state_encoder': state_encoder.state_dict() if state_encoder is not None else None,
                'state_dict_state_decoder': state_decoder.state_dict() if state_decoder is not None else None
            }, filename=args.name)
            with torch.no_grad():
                auc_score = calculate_metrics(test_loader, num_atoms, num_edge_types, num_states, seq_len, encoder, state_encoder, decoder, state_decoder, temperature_state, epoch+1)
            #if auc_score < 0.5 and epoch>=1000:
            #    return
        if epoch > 100 and epoch%10==0:
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
    for i, sample in enumerate(train_loader):
        data, state, graph = sample[0], sample[1], sample[2]
        state = None
        sequence = data.transpose(1,2)
        target = sequence[:, :, 1:, :].float().to(device)
        mask = None if not args.masked else sample[3].to(device)
        # set input-target
        var = Variable(sequence.float(), requires_grad=True).to(device)
        optimizer.zero_grad()
        # compute state probabilities
        if args.encoder !='acd' and args.hidden_states:
            if args.state_decoder=='region':
                state_logits = state_encoder(var[:,:,:,:])
            else:
                state_logits = state_encoder(var[:,:,:,:], rel_rec, rel_send)
            state_samp = simple(state_logits, tau=temperature_state)
            state_prob = my_softmax(state_logits, -1)
            if args.state_decoder=='region':
                state_in = var
                state_prior = None
            else:
                state_in = torch.cat([var[:,:,:-1,:], var[:,:,1:,:], state_samp[:,:,:-1,:]], dim=-1)
                state_prior = state_decoder(state_in)
        elif args.encoder=='acd':
            state_samp = torch.ones_like(var[:,:,:,0:1])
            state_prob = None
        else:
            state_samp = state.float().to(device).transpose(1,2)
            state_prob = None
        # compute edge probabilities
        edge_logits = encoder().unsqueeze(0).repeat(var.size(0),1,1,1)
        # sample using gumble-softmax 
        edges = gumbel_softmax(edge_logits, tau=0.5)
        edge_prob = my_softmax(edge_logits, -1)

        values_out = decoder(var, state_samp, edges, rel_rec, rel_send, args.prediction_steps)
        # record loss
        if args.masked:
            nll = nll_gaussian(values_out*mask[:,None,1:,None].to(device), target, 5e-5)
        else:
            nll = nll_gaussian(values_out, target, 5e-5)
        if args.state_decoder=='region' or args.encoder=='acd' or not args.hidden_states:
            kl = 0
        else:
            kl = kl_categorical(state_prob[:,:,1:], torch.log(state_prior + 1e-16), num_atoms=1)
        for k in range(num_states):
            #if k==0:
            #    kl += kl_categorical(edge_prob[:,:,k,:], torch.log(torch.tensor([0.99, 0.01]).to(device)), num_atoms=1)
            #else:
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
            sys.stdout.flush()
        writer.add_scalar('data/nll_loss', nll, i + epoch*len(train_loader))
        writer.add_scalar('data/kl_loss', kl, i + epoch*len(train_loader))
        writer.add_scalar('data/total_loss', loss, i + epoch*len(train_loader))
        writer.add_scalar('data/mse_loss', mse, i + epoch*len(train_loader))


def calculate_metrics(data_loader, num_atoms, num_edge_types, num_states, seq_len, encoder, state_encoder, decoder, state_decoder, temperature_state, epoch):
    # Calculate metrics from a single batch
    device = args.device
    sample = next(iter(data_loader))
    sequence, state, graph = sample[0], sample[1], sample[2]
    
    
    sequence = sequence.transpose(1,2)
    es = torch.LongTensor(np.array(list(permutations(range(num_atoms), 2))).T).to(device)
    rel_rec, rel_send = create_rel_rec_send(num_atoms)
    rel_rec, rel_send = rel_rec.to(device), rel_send.to(device)
    # set input-target
    var = Variable(sequence.float(), requires_grad=False).to(device)
    target = sequence[:, :, 1:, :].float().to(device)
    values_target = target
    #target_graph = adj_mat_to_target(graph.to(device), rel_rec, rel_send).long()
    # compute edge probabilities
    if args.encoder !='acd' and args.hidden_states:
        if args.state_decoder=='region':
            state_logits = state_encoder(var[:,:,:,:])
        else:
            state_logits = state_encoder(var[:,:,:,:], rel_rec, rel_send)
        state_samp = simple(state_logits, tau=temperature_state)
    elif args.encoder=='acd':
        state_samp = torch.ones_like(var[:,:,:,0:1])
    else:
        state_samp = state.float().to(device).transpose(1,2)
    edge_probs = encoder().unsqueeze(0).repeat(var.size(0),1,1,1)
    # sample using gumble-softmax 
    edges = gumbel_softmax(edge_probs, tau=0.5, hard=True)
    prob = my_softmax(edge_probs, -1)
    # compute decoder predictions
    output = decoder(var, state_samp, edges, rel_rec, rel_send, args.prediction_steps)
    values_out = output[:,:,:,:]
    # MSE positions
    mse_positions = calculate_mse(values_out, values_target)
    writer.add_scalar('metrics/mse_positions', mse_positions, epoch)
    # encoder edge classification accuracy
    graph = graph.to(device)
    pred = prob[...,1].max(-1)[0]
    if args.encoder=='fixed':
        roc_auc = roc_auc_score(graph[0].flatten().detach().cpu().numpy(),pred[0].flatten().detach().cpu().numpy())
        edge_acc = (graph[0].long()==(pred[0]>0.5).long()).float().mean()
    else:
        roc_auc = roc_auc_score(graph.flatten().detach().cpu().numpy(),pred.flatten().detach().cpu().numpy())
        edge_acc = (graph.long()==(pred>0.5).long()).float().mean()
    writer.add_scalar('metrics/encoder_edge_accuracy', edge_acc, epoch)
    writer.add_scalar('metrics/roc_auc_score', roc_auc, epoch)
    if not args.hidden_states:
        if num_atoms==21:
            state_code = torch.tensor([0] + [1 for _ in range(4)] + [0 for _ in range(16)]).to(device).float()
        elif num_atoms==49:
            state_code = torch.tensor([0 for _ in range(5)] + [1 for _ in range(28)] + [0 for _ in range(16)]).to(device).float()
        else:
            state_code = torch.tensor([0 for _ in range(6)] + [1 for _ in range(30)] + [0 for _ in range(16)]).to(device).float()
        senders_states = torch.matmul(rel_send, state_code).long()
        graph_aligned = prob[0,...,1][range(prob.shape[1]),senders_states]
        roc_auc = roc_auc_score(graph[0].flatten().detach().cpu().numpy(),graph_aligned.flatten().detach().cpu().numpy())
        edge_acc = (graph[0].long()==(graph_aligned>0.5).long()).float().mean()
        writer.add_scalar('metrics_obs/encoder_edge_accuracy', edge_acc, epoch)
        writer.add_scalar('metrics_obs/roc_auc_score', roc_auc, epoch)
    
    # decoder state classification accuracy
    print('Epoch: [{0}]\t'
            'RocAUC {roc_auc:.3f}\t'
            'ACC {edge_acc:.4e}\t MSE: {mse_positions:.4e}\t'.format(
            epoch, roc_auc=roc_auc, edge_acc=edge_acc, mse_positions=mse_positions))
    return roc_auc


def save_checkpoint(state, filename='model'):
    os.makedirs("models_ICML_GRN/", exist_ok=True)
    torch.save(state, "models_ICML_GRN/" + filename + '_latest.pth.tar')

if __name__ == '__main__':
    main()
