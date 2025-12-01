"""
Code extracted from https://github.com/loeweX/AmortizedCausalDiscovery (MIT License).
"""

from logging import log
import numpy as np
import torch
import torch.nn.functional as F
import torch.distributions as tdist
from torch.autograd import Variable
from sklearn.metrics import roc_auc_score
from collections import defaultdict
import os

def my_softmax(input, axis=1):
    """Based on https://github.com/ethanfetaya/NRI (MIT License)."""
    trans_input = input.transpose(axis, 0).contiguous()
    soft_max_1d = F.softmax(trans_input, dim=0)
    return soft_max_1d.transpose(axis, 0)


def sample_gumbel(shape, eps=1e-10):
    """
    NOTE: Stolen from https://github.com/pytorch/pytorch/pull/3341/commits/327fcfed4c44c62b208f750058d14d4dc1b9a9d3
    Sample from Gumbel(0, 1)
    based on
    https://github.com/ericjang/gumbel-softmax/blob/3c8584924603869e90ca74ac20a6a03d99a91ef9/Categorical%20VAE.ipynb ,
    (MIT license)
    """
    U = torch.rand(shape).float()
    return -torch.log(eps - torch.log(U + eps))


def gumbel_softmax_sample(logits, tau=1, eps=1e-10, num_samples=1):
    """
    NOTE: Stolen from https://github.com/pytorch/pytorch/pull/3341/commits/327fcfed4c44c62b208f750058d14d4dc1b9a9d3
    Draw a sample from the Gumbel-Softmax distribution
    based on
    https://github.com/ericjang/gumbel-softmax/blob/3c8584924603869e90ca74ac20a6a03d99a91ef9/Categorical%20VAE.ipynb
    (MIT license)
    """
    shape = [num_samples] + list(logits.shape)
    gumbel_noise = sample_gumbel(shape, eps=eps)
    if logits.is_cuda:
        gumbel_noise = gumbel_noise.to(logits.device)
    if num_samples != 1:
        logits = logits.unsqueeze(0)
    else:
        gumbel_noise.squeeze_(0)
    y = logits + Variable(gumbel_noise)
    return my_softmax(y / tau, axis=-1)

def gumbel_softmax(logits, tau=1, hard=False, eps=1e-10):
    """
    NOTE: Stolen from https://github.com/pytorch/pytorch/pull/3341/commits/327fcfed4c44c62b208f750058d14d4dc1b9a9d3
    Sample from the Gumbel-Softmax distribution and optionally discretize.
    Args:
      logits: [batch_size, n_class] unnormalized log-probs
      tau: non-negative scalar temperature
      hard: if True, take argmax, but differentiate w.r.t. soft sample y
    Returns:
      [batch_size, n_class] sample from the Gumbel-Softmax distribution.
      If hard=True, then the returned sample will be one-hot, otherwise it will
      be a probability distribution that sums to 1 across classes
    Constraints:
    - this implementation only works on batch_size x num_features tensor for now
    based on
    https://github.com/ericjang/gumbel-softmax/blob/3c8584924603869e90ca74ac20a6a03d99a91ef9/Categorical%20VAE.ipynb ,
    (MIT license)
    """
    y_soft = gumbel_softmax_sample(logits, tau=tau, eps=eps)
    if hard:
        shape = logits.size()
        _, k = y_soft.data.max(-1)
        # this bit is based on
        # https://discuss.pytorch.org/t/stop-gradients-for-st-gumbel-softmax/530/5
        y_hard = torch.zeros(*shape)
        if y_soft.is_cuda:
            y_hard = y_hard.to(logits.device)
        y_hard = y_hard.zero_().scatter_(-1, k.view(shape[:-1] + (1,)), 1.0)
        # this cool bit of code achieves two things:
        # - makes the output value exactly one-hot (since we add then
        #   subtract y_soft value)
        # - makes the gradient equal to y_soft gradient (since we strip
        #   all other gradients)
        y = Variable(y_hard - y_soft.data) + y_soft
    else:
        y = y_soft
    return y

def simple(logits, tau=1, num_samples=1):
    y = torch.softmax(logits, dim=-1)
    _, k = gumbel_softmax_sample(logits, tau=tau, num_samples=num_samples).data.max(-1)
    shape = ([num_samples] + list(logits.shape)) if num_samples != 1 else list(logits.shape)
    y_hard = torch.zeros(shape).to(logits.device)
    y_hard = y_hard.zero_().scatter_(-1, k.view(y_hard.shape[:-1] + (1,)), 1.0)
    y = Variable(y_hard - y.data) + y
    return y

def encode_onehot(labels):
    """Based on https://github.com/ethanfetaya/NRI (MIT License)."""
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
    return labels_onehot


def kl_categorical(preds, log_prior, num_atoms, eps=1e-16):
    """Based on https://github.com/ethanfetaya/NRI (MIT License)."""
    kl_div = preds * (torch.log(preds + eps) - log_prior)
    return kl_div.sum() / (num_atoms * preds.size(0))

def categorical_log_prob(logits, sample):
    logprob = F.log_softmax(logits, dim=-1)
    return torch.sum(sample*logprob, dim=-1)


def kl_categorical_uniform(
    preds, num_atoms, num_edge_types, add_const=False, eps=1e-16
):
    """Based on https://github.com/ethanfetaya/NRI (MIT License)."""
    kl_div = preds * (torch.log(preds + eps) - np.log(1/num_edge_types + eps))
    if add_const:
        const = np.log(num_edge_types)
        kl_div += const
    return kl_div.sum() / (num_atoms * preds.size(0))


def nll_gaussian(preds, target, variance, add_const=False):
    """Based on https://github.com/ethanfetaya/NRI (MIT License)."""
    neg_log_p = (preds - target) ** 2 / (2 * variance)
    if add_const:
        const = 0.5 * np.log(2 * np.pi * variance)
        neg_log_p += const
    return neg_log_p.sum() / (target.size(0) * target.size(1))


def edge_accuracy(preds, target, binary=True):
    """Based on https://github.com/ethanfetaya/NRI (MIT License)."""
    _, preds = preds.max(-1)
    if binary:
        preds = (preds >= 1).long()
    correct = preds.float().data.eq(target.float().data.view_as(preds)).cpu().sum()
    return float(correct) / (torch.numel(target))


def calc_auroc(pred_edges, GT_edges):
    pred_edges = 1 - pred_edges[:, :, 0]
    return roc_auc_score(
        GT_edges.cpu().detach().flatten(),
        pred_edges.cpu().detach().flatten(),  # [:, :, 1]
    )


def kl_latent(args, prob, log_prior, predicted_atoms):
    if args.prior != 1:
        return kl_categorical(prob, log_prior, predicted_atoms)
    else:
        return kl_categorical_uniform(prob, predicted_atoms, args.edge_types)


def get_observed_relations_idx(num_atoms):
    length = (num_atoms ** 2) - num_atoms * 2
    remove_idx = np.arange(length)[:: num_atoms - 1][1:] - 1
    idx = np.delete(np.linspace(0, length - 1, length), remove_idx)
    return idx


def calculate_mse(output, target, reduce=True):
    if reduce:
        reduction = 'mean'
    else:
        reduction = 'sum'
    mse = F.mse_loss(output, target, reduction=reduction)
    return mse

def calculate_mse_per_sample(output, target):
    mse = F.mse_loss(output, target, reduction='none')
    return mse.mean(dim=(1,2,3))




def edge_accuracy_per_sample(preds, target):
    _, preds = preds.max(-1)
    correct = torch.eq(preds, target)
    b, num_rel, num_states = preds.size()
    return correct.sum(dim=(1,2))/(num_rel*num_states)


def auroc_per_num_influenced(preds, target, total_num_influenced):
    preds = 1 - preds[:, :, 0]
    preds = preds.cpu().detach().numpy()
    target = target.cpu().detach().numpy()

    preds_per_num_influenced = defaultdict(list)
    targets_per_num_influenced = defaultdict(list)

    for idx, k in enumerate(total_num_influenced):
        preds_per_num_influenced[k].append(preds[idx])
        targets_per_num_influenced[k].append(target[idx])

    auc_per_num_influenced = np.zeros((max(preds_per_num_influenced) + 1))
    for num_influenced, elem in preds_per_num_influenced.items():
        auc_per_num_influenced[num_influenced] = roc_auc_score(
            np.vstack(targets_per_num_influenced[num_influenced]).flatten(),
            np.vstack(elem).flatten(),
        )

    return auc_per_num_influenced


def edge_accuracy_observed(preds, target, num_atoms=5):
    idx = get_observed_relations_idx(num_atoms)
    _, preds = preds.max(-1)
    correct = preds[:, idx].eq(target[:, idx]).cpu().sum()
    return np.float(correct) / (target.size(0) * len(idx))


def calc_auroc_observed(pred_edges, GT_edges, num_atoms=5):
    idx = get_observed_relations_idx(num_atoms)
    pred_edges = pred_edges[:, :, 1]
    return roc_auc_score(
        GT_edges[:, idx].cpu().detach().flatten(),
        pred_edges[:, idx].cpu().detach().flatten(),
    )


def kl_normal_reverse(prior_mean, prior_std, mean, log_std, downscale_factor=1):
    std = softplus(log_std) * downscale_factor
    d = tdist.Normal(mean, std)
    prior_normal = tdist.Normal(prior_mean, prior_std)
    return tdist.kl.kl_divergence(d, prior_normal).mean()


def sample_normal_from_latents(latent_means, latent_logsigmas, downscale_factor=1):
    latent_sigmas = softplus(latent_logsigmas) * downscale_factor
    eps = torch.randn_like(latent_sigmas)
    latents = latent_means + eps * latent_sigmas
    return latents


def softplus(x):
    return torch.log(1.0 + torch.exp(x))


def distribute_over_GPUs(args, model, num_GPU=None):
    ## distribute over GPUs
    if args.device.type != "cpu":
        if num_GPU is None:
            model = torch.nn.DataParallel(model)
            num_GPU = torch.cuda.device_count()
            args.batch_size_multiGPU = args.batch_size * num_GPU
        else:
            assert (
                num_GPU <= torch.cuda.device_count()
            ), "You cant use more GPUs than you have."
            model = torch.nn.DataParallel(model, device_ids=list(range(num_GPU)))
            args.batch_size_multiGPU = args.batch_size * num_GPU
    else:
        model = torch.nn.DataParallel(model)
        args.batch_size_multiGPU = args.batch_size

    model = model.to(args.device)

    return model, num_GPU


def create_rel_rec_send(num_atoms, cuda=False):
    """Based on https://github.com/ethanfetaya/NRI (MIT License)."""
    #if args.unobserved > 0 and args.model_unobserved == 1:
    #    num_atoms -= args.unobserved

    # Generate off-diagonal interaction graph
    off_diag = np.ones([num_atoms, num_atoms]) - np.eye(num_atoms)

    rel_rec = np.array(encode_onehot(np.where(off_diag)[0]), dtype=np.float32)
    rel_send = np.array(encode_onehot(np.where(off_diag)[1]), dtype=np.float32)
    rel_rec = torch.FloatTensor(rel_rec)
    rel_send = torch.FloatTensor(rel_send)

    if cuda:
        rel_rec = rel_rec.cuda()
        rel_send = rel_send.cuda()

    return rel_rec, rel_send


def append_losses(losses_list, losses):
    for loss, value in losses.items():
        if type(value) == float:
            losses_list[loss].append(value)
        elif type(value) == defaultdict:
            if losses_list[loss] == []:
                losses_list[loss] = defaultdict(list)
            for idx, elem in value.items():
                losses_list[loss][idx].append(elem)
        else:
            losses_list[loss].append(value.item())
    return losses_list


def average_listdict(listdict, num_atoms):
    average_list = [None] * num_atoms
    for k, v in listdict.items():
        average_list[k] = sum(v) / len(v)
    return average_list


# Latent Temperature Experiment utils
def get_uniform_parameters_from_latents(latent_params):
    n_params = latent_params.shape[1]
    logit_means = latent_params[:, : n_params // 2]
    logit_widths = latent_params[:, n_params // 2 :]
    means = sigmoid(logit_means)
    widths = sigmoid(logit_widths)
    mins, _ = torch.min(torch.cat([means, 1 - means], dim=1), dim=1, keepdim=True)
    widths = mins * widths
    return means, widths


def sigmoid(x):
    return 1.0 / (1.0 + torch.exp(-x))


def sample_uniform_from_latents(latent_means, latent_width):
    latent_dist = tdist.uniform.Uniform(
        latent_means - latent_width, latent_means + latent_width
    )
    latents = latent_dist.rsample()
    return latents


def get_categorical_temperature_prior(mid, num_cats, to_torch=True, to_cuda=True):
    categories = [mid * (2.0 ** c) for c in np.arange(num_cats) - (num_cats // 2)]
    if to_torch:
        categories = torch.Tensor(categories)
    if to_cuda:
        categories = categories.cuda()
    return categories


def kl_uniform(latent_width, prior_width):
    eps = 1e-8
    kl = torch.log(prior_width / (latent_width + eps))
    return kl.mean()


def get_uniform_logprobs(inferred_mu, inferred_width, temperatures):
    latent_dist = tdist.uniform.Uniform(
        inferred_mu - inferred_width, inferred_mu + inferred_width
    )
    cdf = latent_dist.cdf(temperatures)
    log_prob_default = latent_dist.log_prob(inferred_mu)
    probs = torch.where(
        cdf * (1 - cdf) > 0.0, log_prob_default, torch.full(cdf.shape, -8).cuda()
    )
    return probs.mean()


def get_preds_from_uniform(inferred_mu, inferred_width, categorical_temperature_prior):
    categorical_temperature_prior = torch.reshape(
        categorical_temperature_prior, [1, -1]
    )
    preds = (
        (categorical_temperature_prior > inferred_mu - inferred_width)
        * (categorical_temperature_prior < inferred_mu + inferred_width)
    ).double()
    return preds


def get_correlation(a, b):
    numerator = torch.sum((a - a.mean()) * (b - b.mean()))
    denominator = torch.sqrt(torch.sum((a - a.mean()) ** 2)) * torch.sqrt(
        torch.sum((b - b.mean()) ** 2)
    )
    return numerator / denominator


def get_offdiag_indices(num_nodes):
    """Based on https://github.com/ethanfetaya/NRI (MIT License)."""
    ones = torch.ones(num_nodes, num_nodes)
    eye = torch.eye(num_nodes, num_nodes)
    offdiag_indices = (ones - eye).nonzero().t()
    offdiag_indices = offdiag_indices[0] * num_nodes + offdiag_indices[1]
    return offdiag_indices

def calculate_dist_linear_edge_types(decoder_params, gt_params):
    num_edge_types = len(decoder_params)
    distances = []
    for _ in range(num_edge_types):
        distances.append(np.sum(np.abs(decoder_params - gt_params)))
        # shift right
        decoder_params = np.roll(decoder_params, 1)
    distances = np.array(distances)
    permutation = np.argmin(distances)
    dist = distances[permutation]
    return (dist, permutation)

def adj_mat_to_onehot(graph, rel_rec, rel_send, num_edges):
    a = torch.matmul(rel_rec,graph.float())
    edges = torch.diagonal(torch.matmul(a, rel_send.t()), dim1=-2, dim2=-1)
    batch, num_rel = edges.size()
    onehot = torch.zeros((batch, num_rel, num_edges))
    onehot.scatter_(-1, edges.unsqueeze(-1).long(), 1)
    return onehot

def adj_mat_to_target(graph, rel_rec, rel_send):
    a = torch.matmul(rel_rec,graph.float())
    edges = torch.diagonal(torch.matmul(a, rel_send.t()), dim1=-2, dim2=-1)
    return edges

def permute_edges(edges):
    batch, num_rel, num_edges = edges.size()
    temp = torch.zeros(batch, num_rel)
    temp2 = torch.zeros_like(temp)
    for i in range(num_edges-1):
        if i == 0:
            continue
        if i == 1:
            temp = edges[:,:,i+1].clone()
            edges[:,:,i+1] = edges[:,:,i]
        else:
            temp2 = edges[:,:,i+1].clone()
            edges[:,:,i+1] = temp
            temp = temp2
    edges[:,:,1] = temp
    return edges

def state_proportion(data_path):
    states = np.load(data_path).transpose((0,2,1))
    sample = states.reshape((-1, states.shape[-1]))
    print(sample.shape)
    counts_one = 0
    for el in sample:
        states = np.unique(el)
        if len(states) == 1:
            counts_one += 1
    print(counts_one/sample.shape[0])
    

if __name__ == "__main__":
    state_proportion('datasets/box_1_more_freq/state_train_springs5_l10000.npy')