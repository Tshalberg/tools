
import pandas as pd
import numpy as np
import h5py
import time
from loading import load_file, load_folder


fn = "/home/thomas/Documents/master_thesis/fridge/studies/direct_reco/data/direct_reco_fits/minimizer_steps/event_12940_10000_3.hdf5"
# fn = "/home/thomas/Documents/master_thesis/fridge/studies/direct_reco/data/direct_reco_fits/minimizer_steps/event_12940_10000_fit_17.hdf5"
fn = "/home/thomas/Documents/master_thesis/fridge/studies/direct_reco/data/1GeV_Efit_1_100GeV/tables/hdf5/Level7_genie_ic.12640.000000.i3.bz2.hdf5"


# data = load_file(fn, verbose=True, pulse=True)

# print data.keys()

# data = load_file(fn, verbose=True, pulse=False)
# print data.keys()


folder = "/home/thomas/Documents/master_thesis/fridge/studies/direct_reco/data/1GeV_Efit_1_100GeV/tables/hdf5"

data = load_folder(folder, verbose=True)


# t1 = time.time()

# f = h5py.File(fn, 'r')
# keys = list(f.keys())

# print "Loading keys: ", time.time() - t1

# data_dict = dict()

# t_tot  = 0
# times = dict()
# for key in keys:
# 	t1 = time.time()
# 	# Avoid loading __I3Index__
# 	if "__" in key:
# 		continue
# 	# print key
# 	df = pd.read_hdf(fn, key=key)
# 	data_dict[key] = df
# 	t2 = time.time()
# 	tdiff = t2-t1
# 	times[key] = tdiff
# 	t_tot += tdiff

# print "Loading dfs: ", t_tot
# for k, v in times.items():
# 	print k, np.round(v, 2)

# print df
