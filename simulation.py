from numba import cuda
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float32
import numpy as np
import math
import matplotlib.pyplot as plt
import time

from tqdm import tqdm_notebook as tqdm

@cuda.jit
def move(rng_states, start_x, start_y, out_x, out_y, doms, rs, domhits, domhitstimes):
    thread_id = cuda.grid(1)
    
    def rng():
        return xoroshiro128p_uniform_float32(rng_states, thread_id)
    
    x = start_x
    y = start_y
    d = rng()*math.pi*2
    vx = math.cos(d)
    vy = math.sin(d)
    absorbed = False
    time = 0
    while not absorbed:
        if rng() < 0.02:#1:
            d = xoroshiro128p_uniform_float32(rng_states, thread_id)*math.pi*2
            vx = math.cos(d)
            vy = math.sin(d)
        if rng() < 0.02:#05:
            absorbed = True
        x += vx
        y += vy
        for i in range(len(doms)):
            domx = doms[i,0]
            domy = doms[i,1]
            r = rs[i]
            if r >= (math.sqrt((domx-x)**2 + (domy-y)**2)):
                domhits[thread_id, i] += 1
                domhitstimes[thread_id, i] = time
                absorbed = True
        time += 1

    out_x[thread_id] = x
    out_y[thread_id] = y


def simulate(doms, rs, x_start, y_start, blocks, threads_per_block, seed=None, verbose=True):

    N = blocks * threads_per_block


    x_y = []
    t1 = time.time()

    # Set a random seed
    if seed is None:
        ran_seed = np.random.randint(1, 123456)
    else:
        ran_seed = seed

    x_start, y_start = np.float32(x_start), np.float32(y_start)

    # Initialize the random states for the kernel
    rng_states = create_xoroshiro128p_states(N, seed=ran_seed)
    # Create empty arrays for the (x, y) values
    out_x, out_y = np.zeros(N, dtype=np.float32), np.zeros(N, dtype=np.float32)
    # Create empty array for domhits
    domhits = np.zeros((N, len(rs)), dtype=np.int32)
    domhit_times = np.zeros((N, len(rs)), dtype=np.int32)
    # Calculate x, y and domhits
    move[blocks, threads_per_block](rng_states, x_start, y_start, out_x, out_y, doms, rs, domhits, domhit_times)
    # Save the hit information
    domhits = np.sum(domhits, axis=0)
    
    t2 = time.time()
    if verbose:
        print (t2-t1)
    x_y = np.array(x_y)
    domhits = np.array(domhits)

    return domhits, domhit_times, t2-t1


def create_time_bins(dhts_obs, i):
    # Calculate specific time seperated bins for a DOM
    # dependent on some threshold in time between 2 photon hits

    obs = dhts_obs[:,i]
    obs = obs[obs!=0]

    maxt = 100
    uniques = np.unique(obs)
    uniques = uniques[uniques <=maxt]
    bs = []
    b = []
    thrs = 20
    if len(uniques) == 1:
        bs = [[0]]
    else:
        for i in range(len(uniques)-1):
            uc = uniques[i]
            un = uniques[i+1]
            if un - uc <= thrs:
                b.append(i)
            elif un - uc > thrs:
                b.append(i)
                bs.append(b)
                b = []
            if i == len(uniques)-2:
                b.append(i+1)
                bs.append(b)
#    print (bs)
    bins = [0]
    for b in bs:
#        print (b)
        bins.append(uniques[b[0]]-thrs/2)
        bins.append(uniques[b[-1]]+thrs/2)
    bins.append(maxt)
    return bins


def combine_bins(dhts_hypo, dhts_obs, bins, binrange, timebins=False):
    # Bin photons in N = bins timebins, and combine for each DOM
    # dhts_hypo: dom hit times for hypothesis
    # dhts_obs: dom hit times for observation
    
    # Calculate oversampling based on ratio between number of simulated photons
    oversampling = (len(dhts_hypo)/float(len(dhts_obs)))
    mu = []
    obs = []
    for i in range(dhts_hypo.shape[1]):

        if timebins:
            bins = create_time_bins(dhts_obs, i)
            binrange = None

        hypo = dhts_hypo[:,i]
        hypo = hypo[hypo!=0]

        vmu, e = np.histogram(hypo, range=binrange, bins=bins)
        vmu = vmu/oversampling

        data = dhts_obs[:,i]
        data = data[data!=0]
        vobs, e = np.histogram(data, range=binrange, bins=bins)

        mu.append(vmu)
        obs.append(vobs)

    mu = np.concatenate(mu)
    obs = np.concatenate(obs)
    return obs, mu




def LLH_scan(dhts_obs, x_range, y_range, N, bins, binrange, params, timebins=False, deltamu=1e-1):
    from tools.statistics import poisson_llh

    xs = np.linspace(x_range[0], x_range[1], N)
    ys = np.linspace(y_range[0], y_range[1], N)

    X, Y = np.meshgrid(xs, ys)
    Z = np.empty_like(X)

    # for i, j in tqdm(np.ndindex(X.shape)):
    for i in tqdm(range(N)):
        for j in range(N):
            x = X[i, j]
            y = Y[i, j]

            params["x_start"] = x
            params["y_start"] = y

            d, dhts_hypo, t = simulate(**params)

            obs, mu = combine_bins(dhts_hypo, dhts_obs, bins, binrange, timebins=timebins)
            llh = poisson_llh(obs, mu, deltamu=deltamu)
            Z[i, j] = -llh

    return X, Y, Z


