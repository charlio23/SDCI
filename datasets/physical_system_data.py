# Code adapted from https://github.com/loeweX/AmortizedCausalDiscovery/blob/master/codebase/data/synthetic_sim.py (MIT License)
"""Based on https://github.com/ethanfetaya/NRI (MIT License)."""

import numpy as np
import matplotlib.pyplot as plt
import time
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation


class SpringSim(object):
    def __init__(
        self,
        n_balls=5,
        box_size=5.,
        loc_std=0.5,
        vel_norm=0.5,
        interaction_strength=0.1,
        noise_var=0.0,
        n_states=1,
        state_type='indep',
        change_state_freq = 1000
    ):
        self.n_balls = n_balls
        self.box_size = box_size
        self.n_states = n_states
        self.loc_std = loc_std
        self.vel_norm = vel_norm
        self.interaction_strength = interaction_strength
        self.noise_var = noise_var
        self.change_state_freq = change_state_freq
        # State: indep | stoch | dynam | region | collision
        self.state_type = state_type

        self._spring_types = np.array([0.0, 0.5, 1.0])
        self._delta_T = 0.001
        self._max_F = 0.1 / self._delta_T
        self.transition_mat_states = None
        self.state_distrib = None

    def _init_trans_mat(self):
        trans_mat = []
        for k in range(self.n_balls):
            trans_mat_i = np.random.uniform(0,1,(self.n_states, self.n_states))
            trans_mat_i /= trans_mat_i.sum(axis=1, keepdims=True)
            trans_mat.append(trans_mat_i)
        return np.array(trans_mat)

    def _init_state_distrib(self):
        state_distrib = []
        for k in range(self.n_balls):
            state_distrib_i = np.random.uniform(0,1,(1, self.n_states))
            state_distrib_i /= state_distrib_i.sum()
            state_distrib.append(state_distrib_i)
        return np.array(state_distrib)


    def _energy(self, loc, vel, edges):
        # disables division by zero warning, since I fix it with fill_diagonal
        with np.errstate(divide="ignore"):

            K = 0.5 * (vel ** 2).sum()
            U = 0
            for i in range(loc.shape[1]):
                for j in range(loc.shape[1]):
                    if i != j:
                        r = loc[:, i] - loc[:, j]
                        dist = np.sqrt((r ** 2).sum())
                        U += (
                            0.5
                            * self.interaction_strength
                            * edges[i, j]
                            * (dist ** 2)
                            / 2
                        )
            return U + K

    def _clamp(self, loc, vel, state):
        """
        :param loc: 2xN location at one time stamp
        :param vel: 2xN velocity at one time stamp
        :return: location and velocity after hitting walls and returning after
            elastically colliding with walls
        """
        assert np.all(loc < self.box_size * 3)
        assert np.all(loc > -self.box_size * 3)

        over = loc > self.box_size
        loc[over] = 2 * self.box_size - loc[over]
        assert np.all(loc <= self.box_size)

        # assert(np.all(vel[over]>0))
        vel[over] = -np.abs(vel[over])
        if self.state_type == 'collision':
            over_reduced= np.any(over, axis=0)
            state[over_reduced] += 1

        under = loc < -self.box_size
        loc[under] = -2 * self.box_size - loc[under]
        # assert (np.all(vel[under] < 0))
        assert np.all(loc >= -self.box_size)
        vel[under] = np.abs(vel[under])
        if self.state_type == 'collision':
            under_reduced= np.any(under, axis=0)
            state[under_reduced] += 1
            state %= self.n_states
            assert np.all(state < self.n_states)
        
        if self.state_type == 'region':
            left = loc[0,:] > 0
            state = left.astype(int)

        return loc, vel, state

    def _l2(self, A, B):
        """
        Input: A is a Nxd matrix
               B is a Mxd matirx
        Output: dist is a NxM matrix where dist[i,j] is the square norm
            between A[i,:] and B[j,:]
        i.e. dist[i,j] = ||A[i,:]-B[j,:]||^2
        """
        A_norm = (A ** 2).sum(axis=1).reshape(A.shape[0], 1)
        B_norm = (B ** 2).sum(axis=1).reshape(1, B.shape[0])
        dist = A_norm + B_norm - 2 * A.dot(B.transpose())
        return dist

    def sample_edges(self, spring_prob, num_max_edges):
        # Sample edges
        """ Method returns edges as follows:
            Set of K matrices, where each row denotes the causal effects.
            For query, each state selects columns from set of K.
            Example of query: CSG for K=2
                [[[0. 0. 0. 0. 0.]
                [0. 0. 0. 0. 0.]
                [0. 0. 0. 0. 0.]
                [0. 0. 0. 0. 0.]
                [0. 0. 0. 0. 0.]]

                [[0. 0. 1. 1. 0.]
                [1. 0. 1. 1. 0.]
                [0. 1. 0. 0. 1.]
                [0. 0. 0. 0. 0.]
                [0. 0. 0. 0. 0.]]]
            For s= [0 1 0 1 0], the graph is:
            [[0. 0. 0. 1. 0.]
             [0. 0. 0. 1. 0.]
             [0. 1. 0. 0. 0.]
             [0. 0. 0. 0. 0.]
             [0. 0. 0. 0. 0.]]
        """
        if (num_max_edges is None) or (num_max_edges==-1):
            edges = np.random.choice(
                self._spring_types, size=(self.n_states, self.n_balls, self.n_balls), p=spring_prob
            )
        else:
            edges = np.zeros((self.n_states, self.n_balls, self.n_balls))
            for idx in range(1, self.n_states):
                for i in range(self.n_balls):
                    edges[idx,:,i][np.random.choice(self.n_balls, size=num_max_edges, replace=False)] = 1
        for i in range(self.n_states):
            np.fill_diagonal(edges[i], 0)
        return edges

    def get_edges(
        self,
        undirected,
        influencer,
        uninfluenced,
        confounder,
        spring_prob=[1 / 2, 0, 1 / 2],
        num_max_edges=None
    ):
        edges = self.sample_edges(spring_prob,num_max_edges)

        if undirected:
            edges = (
                np.tril(edges) + np.tril(edges, -1).T
            )  # this is commented out to get directed graphs

        if confounder:
            while sum(edges[:, -1]) < 2:
                edges = self.sample_edges(spring_prob)

        if influencer:
            edges[:, -1] = 1

        if uninfluenced:
            edges[-1, :] = 0

            # print(loc_fixed, vel_fixed)
        return edges
    
    def sample_state(self):
        state_next = []
        for k in range(self.n_balls):
            state_next.append(np.random.choice(self.n_states, p=self.state_distrib[k].reshape(-1)))
        return np.array(state_next)

    def sample_trajectory(
        self,
        T=10000,
        sample_freq=10,
        spring_prob=[0.5, 0, 0.5],
        #spring_prob=[0.85, 0, 0.15],
        undirected=False,
        fixed_particle=False,
        influencer=False,
        uninfluenced=False,
        confounder=False,
        edges=None,
        num_max_edges = None,
    ):
        n = self.n_balls
        assert T % sample_freq == 0
        T_save = int(T / sample_freq - 1)
        diag_mask = np.ones((n, n), dtype=bool)
        np.fill_diagonal(diag_mask, 0)
        counter = 0

        if edges is None:
            edges = self.get_edges(
                undirected=undirected,
                influencer=influencer,
                uninfluenced=uninfluenced,
                confounder=confounder,
                spring_prob=spring_prob,
                num_max_edges=num_max_edges
            )

        # Initialize location, velocity and state
        loc = np.zeros((T_save, 2, n))
        vel = np.zeros((T_save, 2, n))
        state = np.zeros((T_save, n))
        loc_next = np.random.randn(2, n) * self.loc_std
        vel_next = np.random.randn(2, n)
        if self.state_type == 'stoch':
            self.transition_mat_states = self._init_trans_mat()
            self.state_distrib = self._init_state_distrib()
            state_next = self.sample_state()
        elif self.state_type=='region':
            state_next = (loc_next[0,:] > 0).astype(int)
        else:
            state_next = np.array([0]*n)
        v_norm = np.sqrt((vel_next ** 2).sum(axis=0)).reshape(1, -1)
        vel_next = vel_next * self.vel_norm / v_norm

        if fixed_particle:
            loc_fixed, vel_fixed, state_fixed = self._clamp(np.random.randn(2), np.zeros(2), np.zeros(1))
            loc_next[:, -1] = loc_fixed
            vel_next[:, -1] = vel_fixed
            state_fixed[:, -1] = state_fixed

        loc[0, :, :], vel[0, :, :], state[0, :] = self._clamp(loc_next, vel_next, state_next)

        # disables division by zero warning, since I fix it with fill_diagonal
        with np.errstate(divide="ignore"):
            #print(edges)
            state_next = np.array([0,1,0,1,0])
            forces_size = -self.interaction_strength * edges.transpose(0,2,1)[state_next, range(self.n_balls)].T
            np.fill_diagonal(
                forces_size, 0
            )  # self forces are zero (fixes division by zero)
            print(forces_size)
            print(state_next, loc_next[0,:])
            print(forces_size.reshape(n, n)*np.subtract.outer(loc_next[0, :], loc_next[0, :]).reshape(
                            n, n))
            print(np.subtract.outer(loc_next[0, :], loc_next[0, :]).reshape(
                            n, n))
            F = (
                forces_size.reshape(1, n, n)
                * np.concatenate(
                    (
                        np.subtract.outer(loc_next[0, :], loc_next[0, :]).reshape(
                            1, n, n
                        ),
                        np.subtract.outer(loc_next[1, :], loc_next[1, :]).reshape(
                            1, n, n
                        ),
                    )
                )
            ).sum(
                axis=-1
            )  # sum over influence from different particles to get their joint force
            print(F[0])
            F[F > self._max_F] = self._max_F
            F[F < -self._max_F] = -self._max_F
            exit()
            vel_next += self._delta_T * F
            # run leapfrog
            for i in range(1, T):
                loc_next += self._delta_T * vel_next
                loc_next, vel_next, state_next = self._clamp(loc_next, vel_next, state_next)

                if fixed_particle:
                    loc_next[:, -1] = loc_fixed
                    vel_next[:, -1] = vel_fixed

                if i % sample_freq == 0:
                    if self.state_type == 'dynam':
                    # Change state if you cross x==0
                        state_change = np.logical_xor((loc[counter-1, 0, :] > 0), (loc_next[0] > 0))
                        state_next += state_change.astype(int)
                        state_next %= self.n_states
                    loc[counter, :, :], vel[counter, :, :], state[counter, :] = loc_next, vel_next, state_next
                    counter += 1

                forces_size = -self.interaction_strength * edges.transpose(0,2,1)[state_next, range(self.n_balls)].T
                np.fill_diagonal(forces_size, 0)
                # assert (np.abs(forces_size[diag_mask]).min() > 1e-10)

                F = (
                    forces_size.reshape(1, n, n)
                    * np.concatenate(
                        (
                            np.subtract.outer(loc_next[0, :], loc_next[0, :]).reshape(
                                1, n, n
                            ),
                            np.subtract.outer(loc_next[1, :], loc_next[1, :]).reshape(
                                1, n, n
                            ),
                        )
                    )
                ).sum(axis=-1)
                F[F > self._max_F] = self._max_F
                F[F < -self._max_F] = -self._max_F
                vel_next += self._delta_T * F
                if self.state_type == 'indep':
                    if i % self.change_state_freq == 0:
                        state_next += 1
                        state_next %= self.n_states
                        assert np.all(state_next < self.n_states)
                elif self.state_type == 'stoch':
                    if i % sample_freq == 0:
                        for k in range(self.n_balls):
                            self.state_distrib[k] = np.dot(self.state_distrib[k], self.transition_mat_states[k])
                        state_next = self.sample_state()


                    

            # Add noise to observations
            loc += np.random.randn(T_save, 2, self.n_balls) * self.noise_var
            vel += np.random.randn(T_save, 2, self.n_balls) * self.noise_var
            return loc, vel, state, edges


def init():
    # init lines
    for line in lines:
        line.set_data([], [])

    # ax.set_xlim([-5.0, 5.0])
    # ax.set_ylim([-5.0, 5.0])
    return lines


def update(frame):
    for j, line in enumerate(lines):
        line.set_data(loc[frame, 0, j], loc[frame, 1, j])
    return lines


if __name__ == "__main__":
    sim = SpringSim(interaction_strength=0.1, n_balls=5, n_states=2, box_size=1., state_type='dynam')

    t = time.time()
    loc, vel, state, edges = sim.sample_trajectory(
        T=8100,
        sample_freq=100,
        fixed_particle=False,
        influencer=False,
        uninfluenced=False,
        confounder=False,
    )  # 5000, 100
    print(edges)
    print("Simulation time: {}".format(time.time() - t))
    sample = state.transpose((1,0))
    #print(np.concatenate([loc[:,0:1,:],state[:,None,:]],1))
    print(state)
    counts_one = 0
    state_freq = []
    for el in sample:
        states = np.unique(el)
        print(states)
        if len(states) == 1:
            counts_one += 1
        state_freq.append(el.mean())
    #print(counts_one/sample.shape[0])
    print(state_freq)
    vel_norm = np.sqrt((vel ** 2).sum(axis=1))
    plt.figure()
    axes = plt.gca()
    axes.set_xlim([-5.0, 5.0])
    axes.set_ylim([-5.0, 5.0])
    for i in range(loc.shape[-1]):
        plt.plot(loc[:, 0, i], loc[:, 1, i])
        plt.plot(loc[0, 0, i], loc[0, 1, i], "d")

    plt.show()

    # Create animation.
    # Set up formatting for the movie files.
    Writer = animation.writers["ffmpeg"]
    writer = Writer(fps=15, metadata=dict(artist="Me"), bitrate=1800)

    fig = plt.figure()
    ax = plt.axes(xlim=(-5, 5), ylim=(-5, 5))

    lines = [
        plt.plot([], [], marker="$" + "{:d}".format(i) + "$", alpha=1, markersize=10)[0]
        for i in range(loc.shape[-1])
    ]
    
    ani = FuncAnimation(fig, update, frames=loc.shape[0], interval=20, blit=True)
    plt.show()
    ani.save(filename="test.mp4", writer=writer)
