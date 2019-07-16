
import pandas as pd
import numpy as np
import time
import os
import re
from tqdm import tqdm_notebook as tqdm
os.environ["TF_CPP_MIN_LOG_LEVEL"]="3"
import h5py

def load_file(fn, specific_keys=None, pulse=False, verbose=False):
    if verbose:
        print "Trying to load file: ", fn
    if specific_keys:
        # Only load specific keys if desired (to save time/memory)
        keys = specific_keys
    else:
        # Get all available keys from hdf5 file
        f = h5py.File(fn, 'r')
        keys = list(f.keys())

        # Filter out unwanted keys like __I3Index__
        if pulse:
            keys = [k for k in keys if "__" not in k]
        else:
            keys = [k for k in keys if "__" not in k and "MonopodFitFinal_fit" not in k]
    # print keys
    data = dict()

    t_tot  = 0
    times = dict()
    for key in keys:
        # print key
        try:
            t1 = time.time()
            df = pd.read_hdf(fn, key=key, mode="r")
            data[key] = df
            t2 = time.time()
            tdiff = t2-t1
            times[key] = tdiff
            t_tot += tdiff
        except:
            print "Error with key: ", key

    # Check if all keys have been loaded
    assert set(keys) == set(data.keys()), "Could not find all expected tables, missing %s" % (set(keys)-set(data.keys()))

    if verbose:
        # Report number of events in file
        num_events = len(data["MCNeutrino"])
        print("Loaded data (%i events)" % num_events)
        print("Total time: %s" % (t_tot))
    return data


def sort_files(files, mode=0):
    files = np.array(files)
    if mode == 0:
        numbs = np.array([re.findall(r"000\d*", f)[0] for f in files], dtype=int)
    elif mode == 1:
        numbs = np.array([re.findall(r"(?<=_fit_0_)\d*", f)[0] for f in files], dtype=int)
    elif mode == 2:
        numbs = np.array([re.findall(r"(?<=_fit_)\d*", f)[0] for f in files], dtype=int)
    elif mode == 3:
        numbs = np.array([re.findall(r"(?<=_0_)\d*", f)[0] for f in files], dtype=int)
    elif type(mode) == str:
        numbs = np.array([re.findall(mode, f)[0] for f in files], dtype=int)
    order = np.argsort(numbs)
    files = files[order]
    return files

def load_folder(folder, specific_keys=None, pulse=False, verbose=False, mode=None):
    files = os.listdir(folder)
    if "Level7" in files[0]:
        # Sorted files if "Level7" in names 
        files = sort_files(files, mode=0)
    elif "fit_0_" in files[0]:
        files = sort_files(files, mode=1)
    elif "_fit_" in files[0]:
        files = sort_files(files, mode=2)
    elif mode is not None:
        files = sort_files(files, mode=mode)
    else:
        files = sort_files(files, mode=3)

    all_data =[]
    err_files = []
    c = 0
    err = 0
    for i in tqdm(range(len(files))):
        f = files[i]
        full_path = folder + "/" + f
        try:
            fit_data = load_file(full_path, specific_keys, pulse, verbose)
            all_data.append(fit_data)
        except Exception as e:
            err += 1
            err_files.append(f)
        c += 1

    print "Loaded {} out of {} files".format(c-err, c)
    if err > 0:
        print "Error loading following files: "
        for f in err_files:
            print f

    return all_data


def load_multiple_folders(folders, specific_keys=None, pulse=False, verbose=False):
    data = dict()
    # Loop through folders
    for label in folders.keys():
        folder = folders[label]
        data[label] = load_folder(folder, specific_keys, pulse, verbose)
        print "Loaded folder: ", label, "\n"

    return data


def parse_outputfile(fn):
    assert (".out" in fn)
    ids_str = ""
    ids = []
    positions_str = ""
    positions = []
    info_str = ""
    infos = []
    kpos = "positions: "
    kid = "string_dom: "
    kinfo = "info: "
    with open(fn, "rb") as f:
        for line in f.readlines():
            if kpos in line:
                positions_str = line.replace(kpos, "")
                positions.append(np.fromstring(positions_str, sep=" "))
                
            if kid in line:
                ids_str = line.replace(kid, "")
                ids.append(np.fromstring(ids_str, sep=" "))
                
            if kinfo in line:
                info_str = line.replace(kinfo, "")
                infos.append(np.fromstring(info_str, sep=" "))

    positions = np.array(positions)
    ids = np.array(ids)
    infos = np.array(infos)

    return positions, ids, infos


def load_I3Geometry(fn):
    from icecube import dataio, dataclasses, icetray

    f = dataio.I3File(fn)
    frame = f.pop_frame()

    geo = frame["I3Geometry"]
    modulegeo = frame["I3ModuleGeoMap"]

    return geo.omgeo, modulegeo