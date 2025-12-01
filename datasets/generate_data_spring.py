"""Based on https://github.com/ethanfetaya/NRI (MIT License)."""

import os
import json
import time
import sys
import numpy as np
import argparse
from scipy.io import savemat

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tqdm import tqdm
from model import utils 
from datasets.physical_system_data import SpringSim


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--simulation", type=str, default="springs", help="What simulation to generate."
    )
    parser.add_argument(
        "--state-type", type=str, default="indep", help="What type of state evolution (indep|stoch|dynam|region)."
    )
    parser.add_argument(
        "--num-max-edges",
        type=int,
        default=-1,
        help="Number of maximum interactions allowed per particle.",
    )
    parser.add_argument(
        "--num-train",
        type=int,
        default=50000,
        help="Number of training simulations to generate.",
    )
    parser.add_argument(
        "--num-valid",
        type=int,
        default=10000,
        help="Number of validation simulations to generate.",
    )
    parser.add_argument(
        "--num-test",
        type=int,
        default=10000,
        help="Number of test simulations to generate.",
    )
    parser.add_argument(
        "--length", type=int, default=5000, help="Length of trajectory."
    )
    parser.add_argument(
        "--box-size", type=float, default=5., help="Size of box."
    )
    parser.add_argument(
        "--sample_freq",
        type=int,
        default=100,
        help="How often to sample the trajectory.",
    )
    parser.add_argument(
        "--n_balls", type=int, default=5, help="Number of balls in the simulation."
    )
    parser.add_argument(
        "--n_edge_types", type=int, default=2, help="Number of edge types in the simulation."
    )
    parser.add_argument(
        "--n_states", type=int, default=1, help="Number of states in the simulation."
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--datadir", type=str, default="data", help="Name of directory to save data to."
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="Temperature of SpringSim simulation.",
    )
    parser.add_argument(
        "--sparsity",
        type=float,
        default=0.5,
        help="Sparsity of SpringSim simulation.",
    )
    parser.add_argument(
        "--temperature_dist",
        action="store_true",
        default=False,
        help="Generate with a varying latent temperature from a categorical distribution.",
    )
    parser.add_argument(
        '--temperature_alpha', 
        type=float, 
        default=None,
        help='middle category of categorical temperature distribution.'
    )
    parser.add_argument(
        '--temperature_num_cats', 
        type=int, 
        default=None,
        help='number of categories of categorical temperature distribution.'
    )
    parser.add_argument(
        "--undirected",
        action="store_true",
        default=False,
        help="Have symmetric connections (non-causal)",
    )
    parser.add_argument(
        "--fixed_particle",
        action="store_true",
        default=False,
        help="Have one particle fixed in place and influence all others",
    )
    parser.add_argument(
        "--influencer_particle",
        action="store_true",
        default=False,
        help="Unobserved particle (last one) influences all",
    )
    parser.add_argument(
        "--confounder",
        action="store_true",
        default=False,
        help="Unobserved particle (last one) influences at least two others",
    )
    parser.add_argument(
        "--uninfluenced_particle",
        action="store_true",
        default=False,
        help="Unobserved particle (last one) is not influence by others",
    )
    parser.add_argument(
        "--fixed_connectivity",
        action="store_true",
        default=False,
        help="Have one inherent causal structure for ALL simulations",
    )
    parser.add_argument(
        '--save-mat', action="store_true", default=False, help='Save data to mat file.'
    )

    args = parser.parse_args()

    if args.fixed_particle:
        args.influencer_particle = True
        args.uninfluenced_particle = True

    assert not (args.confounder and args.influencer_particle), "These options are mutually exclusive."

    args.length_test = args.length * 2

    print(args)
    return args

def generate_dataset(num_sims, length, sample_freq, edges=None, sampled_sims=None):
    if not sampled_sims is None:
        assert len(sampled_sims) == num_sims

    loc_all = list()
    vel_all = list()
    state_all = list()
    edges_all = list()
    counter = 0
    if args.state_type == 'dynam' or args.state_type == 'region' or args.state_type =='collision':
        with tqdm(total=num_sims) as pbar:
            while(counter < num_sims):
                loc, vel, state, edges = sim.sample_trajectory(
                    T=length,
                    sample_freq=sample_freq,
                    undirected=args.undirected,
                    fixed_particle=args.fixed_particle,
                    influencer=args.influencer_particle,
                    uninfluenced=args.uninfluenced_particle,
                    confounder=args.confounder,
                    edges=edges,
                    num_max_edges=args.num_max_edges
                )
                T = length//sample_freq-1
                if T > 120:
                    T //= 2
                counts_bad = 0
                for el in state[:T,:].transpose((1,0)):
                    #states = np.sum(el)
                    #print(el)
                    #prop = states/T
                    #print(prop)
                    #if not(prop > 0.2 and prop < 0.8):
                    #    counts_bad += 1
                    _, counts = np.unique(el, return_counts=True)
                    prop = counts/T
                    if (prop.shape[0] != args.n_states or (prop < 0.1).any() or ((prop < 0.05).any() and args.num_states==4) or (el[:5]==1).any()):
                        counts_bad += 1
                prop = counts_bad/state.shape[1]
                if prop <= 1e-3 or (prop <=0.25 and args.fixed_connectivity) or (prop <=0.2 and args.n_balls>=10 and args.state_type=='collision'):
                    loc_all.append(loc)
                    vel_all.append(vel)
                    state_all.append(state)
                    edges_all.append(edges)
                    counter += 1
                    pbar.update(1)
                if not args.fixed_connectivity:
                    edges = None
            
    else:
        for i in tqdm(range(num_sims)):
            if not sampled_sims is None:
                sim_i = sampled_sims[i]
            else:
                sim_i = sim
            loc, vel, state, edges = sim_i.sample_trajectory(
                T=length,
                sample_freq=sample_freq,
                undirected=args.undirected,
                fixed_particle=args.fixed_particle,
                influencer=args.influencer_particle,
                uninfluenced=args.uninfluenced_particle,
                confounder=args.confounder,
                edges=edges
            )
            loc_all.append(loc)
            vel_all.append(vel)
            state_all.append(state)
            edges_all.append(edges)

            if not args.fixed_connectivity:
                edges = None

    loc_all = np.stack(loc_all)
    vel_all = np.stack(vel_all)
    state_all = np.stack(state_all)
    edges_all = np.stack(edges_all)

    return loc_all, vel_all, state_all, edges_all


if __name__ == "__main__":

    args = parse_args()
    if args.temperature_dist:
        categories = utils.get_categorical_temperature_prior(
                args.temperature_alpha, 
                args.temperature_num_cats, 
                to_torch=False, 
                to_cuda=False
            )
        print('Drawing from uniform categorical distribution over: ', categories)

    if args.simulation == "springs":
        if not args.temperature_dist:
            sim = SpringSim(
                noise_var=0.0,
                n_balls=args.n_balls,
                interaction_strength=args.temperature,
                num_edge_types=args.n_edge_types,
                n_states=args.n_states,
                box_size=args.box_size,
                state_type=args.state_type
            )
        else:
            temperature_samples_train = np.random.choice(categories, size=args.num_train)
            sims_train = [
                SpringSim(noise_var=0.0, n_balls=args.n_balls, interaction_strength=t)
                for t in temperature_samples_train
            ]
            temperature_samples_valid = np.random.choice(categories, size=args.num_valid)
            sims_valid = [
                SpringSim(noise_var=0.0, n_balls=args.n_balls, interaction_strength=t)
                for t in temperature_samples_valid
            ]
            temperature_samples_test = np.random.choice(categories, size=args.num_test)
            sims_test = [
                SpringSim(noise_var=0.0, n_balls=args.n_balls, interaction_strength=t)
                for t in temperature_samples_test
            ]
    else:
        raise ValueError("Simulation {} not implemented".format(args.simulation))

    suffix = "_" + args.simulation

    suffix += str(args.n_balls)

    if args.undirected:
        suffix += "undir"

    if args.fixed_particle:
        suffix += "_fixed"

    if args.uninfluenced_particle:
        suffix += "_uninfluenced"

    if args.influencer_particle:
        suffix += "_influencer"

    if args.confounder:
        suffix += "_conf"

    if args.temperature != 0.1:
        suffix += "_inter" + str(args.temperature)

    if args.length != 5000:
        suffix += "_l" + str(args.length)

    if args.num_train != 50000:
        suffix += "_s" + str(args.num_train)

    if args.fixed_connectivity:
        suffix += "_oneconnect"

    print(suffix)

    np.random.seed(args.seed)
    if args.fixed_connectivity:
        edges = sim.get_edges(
            undirected=args.undirected,
            influencer=args.influencer_particle,
            uninfluenced=args.uninfluenced_particle,
            confounder=args.confounder,
            spring_prob=[args.sparsity, 0, 1-args.sparsity],
            num_max_edges=args.num_max_edges
        )
        print('\x1b[5;30;41m' + "Edges are fixed to be: " + '\x1b[0m')
        print(edges)
    else:
        edges = None
    print("Generating {} training simulations".format(args.num_train))
    loc_train, vel_train, state_train, edges_train = generate_dataset(
        args.num_train,
        args.length,
        args.sample_freq,
        edges,
        sampled_sims=(None if not args.temperature_dist else sims_train),
    )

    print("Generating {} validation simulations".format(args.num_valid))
    loc_valid, vel_valid, state_valid, edges_valid = generate_dataset(
        args.num_valid,
        args.length,
        args.sample_freq,
        edges,
        sampled_sims=(None if not args.temperature_dist else sims_valid),
    )

    print("Generating {} test simulations".format(args.num_test))
    loc_test, vel_test, state_test, edges_test = generate_dataset(
        args.num_test,
        args.length_test,
        args.sample_freq,
        edges,
        sampled_sims=(None if not args.temperature_dist else sims_test),
    )

    if not os.path.exists(args.datadir):
        os.makedirs(args.datadir)

    json.dump(
        vars(args),
        open(os.path.join(args.datadir, "args.json"), "w"),
        indent=4,
        separators=(",", ": "),
    )
    np.save(os.path.join(args.datadir, "loc_train" + suffix + ".npy"), loc_train)
    np.save(os.path.join(args.datadir, "vel_train" + suffix + ".npy"), vel_train)
    np.save(os.path.join(args.datadir, "state_train" + suffix + ".npy"), state_train)
    np.save(os.path.join(args.datadir, "edges_train" + suffix + ".npy"), edges_train)

    np.save(os.path.join(args.datadir, "loc_valid" + suffix + ".npy"), loc_valid)
    np.save(os.path.join(args.datadir, "vel_valid" + suffix + ".npy"), vel_valid)
    np.save(os.path.join(args.datadir, "state_valid" + suffix + ".npy"), state_valid)
    np.save(os.path.join(args.datadir, "edges_valid" + suffix + ".npy"), edges_valid)

    np.save(os.path.join(args.datadir, "loc_test" + suffix + ".npy"), loc_test)
    np.save(os.path.join(args.datadir, "vel_test" + suffix + ".npy"), vel_test)
    np.save(os.path.join(args.datadir, "state_test" + suffix + ".npy"), state_test)
    np.save(os.path.join(args.datadir, "edges_test" + suffix + ".npy"), edges_test)

    if args.save_mat:
        save_res = {
            'loc_save': loc_test,
            'vel_save': vel_test,
            'state_save': state_test,
            'G_save': edges_test
        }
        savemat(args.datadir, save_res)

    if args.temperature_dist:
        np.save(
            os.path.join(args.datadir, "temperatures_train" + suffix + ".npy"),
            temperature_samples_train,
        )
        np.save(
            os.path.join(args.datadir, "temperatures_valid" + suffix + ".npy"),
            temperature_samples_valid,
        )
        np.save(
            os.path.join(args.datadir, "temperatures_test" + suffix + ".npy"),
            temperature_samples_test,
        )