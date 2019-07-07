import numpy as np
import pandas as pd
from tqdm import tqdm_notebook as tqdm

# Extract all values from the 7 reconstruction parameters
def extract_7D(all_data, seed_key, N=None):
    params = ["x", "y", "z", "time", "zenith", "azimuth", "energy", "Event"]
    params_true = {"x":[], "y":[], "z":[], "time":[], "zenith":[], "azimuth":[], "energy":[], "Event":[]}
    params_fit = {"x":[], "y":[], "z":[], "time":[], "zenith":[], "azimuth":[], "energy":[], "Event":[]}
    params_seed = {"x":[], "y":[], "z":[], "time":[], "zenith":[], "azimuth":[], "energy":[], "Event":[]}
    params_best = {"x":[], "y":[], "z":[], "time":[], "zenith":[], "azimuth":[], "energy":[], "Event":[]}

    if N:
        assert len(all_data) >= N
    else:
        N = len(all_data)

    for i in range(N):
        events_data = all_data[i]

        true = events_data["MCNeutrino"]
        fit = events_data["MonopodFitFinal"]
        best = events_data["Monopod_best"]
        seed = events_data[seed_key]
        for p in params:
            v_seed = seed[p]
            v_best = best[p]
            v_fit = fit[p]
            v_true = true[p]

            params_true[p].append(v_true)
            params_fit[p].append(v_fit)
            params_seed[p].append(v_seed)
            params_best[p].append(v_best)
    
    for p in params:
        params_true[p] = np.concatenate(params_true[p])
        params_fit[p] = np.concatenate(params_fit[p])
        params_seed[p] = np.concatenate(params_seed[p])
        params_best[p] = np.concatenate(params_best[p])

    return pd.DataFrame(params_true), pd.DataFrame(params_fit), pd.DataFrame(params_seed), pd.DataFrame(params_best)


def extract_fits(data, N=None):
    seed_dict = {0:"TruthSeed", 1:"FixedSed", 2:"Monopod_best", 3:"SPEFit32"}

    fits = dict()
    labels = data.keys()
    for label in labels:
        data_in = data[label]
        if data_in[0]["InfoGeneral"].fit.values[0] == 1:
            seed_key = seed_dict[data_in[0]["InfoGeneral"].seed.values[0]]
            params_true, params_fit, params_seed, params_best = extract_7D(data_in, seed_key, N)
            fits[label] = params_fit
    fits["best"] = params_best
    fits["true"] = params_true

    #TODO: Make consistency checks for all extracted fits (same number events etc.)

    return fits


def sort_by_oversampling(data):
    data_event = dict()
    oss = set()
    for d in data:
        oversampling = int(d["InfoGeneral"]["Oversampling"].values[0])
        if oversampling in data_event:
            data_event[oversampling].append(d)
        else:
            data_event[oversampling] = [d]
        oss.add(oversampling)
    oss = sorted(list(oss))

    data_all = []
    for oversampling in oss:
        data_all.append(data_event[oversampling])
        
    return data_all, oss