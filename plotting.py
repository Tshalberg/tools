
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# from utils.plotting.standard_modules import *
# from utils.plotting.percentile import plot_y_percentiles 
from matplotlib.backends.backend_pdf import PdfPages
import os

# plt.style.use('ggplot')
# mpl.rcParams['xtick.labelsize'] = 25
# mpl.rcParams['ytick.labelsize'] = 25
# mpl.rcParams['axes.labelsize'] = 25
# mpl.rcParams['legend.fontsize'] = 15
# mpl.rcParams['font.size'] = 15
# np.set_printoptions(suppress=True)

def calc_bands(fit, conf=90):
    # print 1-conf
    lower = np.percentile(fit, (100-conf)/2)
    mid = np.percentile(fit, 50)
    upper = np.percentile(fit, conf + (100-conf)/2)
    return lower, mid, upper

def plot_bands(fit, ax, color=None, lw=2):
    lower, mid, upper = calc_bands(fit)
    # print "%.7s, %.7s, %.7s" % (lower, mid, upper)
    if color is None:
        ax.axvline(lower, 0, 1, ls="--", lw=lw, label="1$\sigma$ [%.5s, %.5s, %.5s]" % (lower, upper, upper-lower), c="k")
        ax.axvline(mid, 0, 1, ls="-", lw=lw, label="mean: %.5s" % mid, c="k")
        ax.axvline(upper, 0, 1, ls="--", lw=lw, label="", c="k")
    else:
        ax.axvline(lower, 0, 1, ls="--", lw=lw, label="1$\sigma$ [%.5s, %.5s, %.5s]" % (lower, upper, upper-lower), c=color)
        ax.axvline(mid, 0, 1, ls="-", lw=lw, label="mean: %.5s" % mid, c=color)
        ax.axvline(upper, 0, 1, ls="--", lw=lw, label="", c=color)       

def plot_1D_compare(fits, labels, bins_dict, pdf_name="1Dhist_plot", Nbins=100, params_print=None):

    folder = "plots/1d_hist/"+"/".join(pdf_name.split("/")[:-1])
    if not os.path.exists(folder):
        os.makedirs(folder)

    label_ref = labels[0]
    for label in labels:
        assert sum(fits[label_ref].Event == fits[label].Event) == len(fits[label_ref].Event)
    
    lw = 3
    # pdf = PdfPages('plots/%s.pdf' % pdf_name)
    params = ["x", "y", "z", "time", "azimuth", "zenith", "energy"]
    for p in params:
        binrange =  (bins_dict[p][0], bins_dict[p][1])
        fig, ax = plt.subplots(figsize=(20,10))
        true = fits["true"]
        for label in labels:
            params_fit = fits[label]
            if p == "energy":
                fit = (params_fit[p]-true[p])/true[p]*100
            else:
                fit = params_fit[p]-true[p]
            plot = ax.hist(fit, range=binrange, bins=Nbins, histtype="step", lw=lw, label=label, alpha=0.75)
            c = list(plot[-1][0].get_facecolor())
            c[-1] = 1
            c = tuple(c)
#             c = "k"
            plot_bands(fit, ax, color=c)
            # lower, mid, upper = calc_bands(fit)
#             print lower, mid, upper
#             ax.axvline(0, 0, 1, color="k")
#             print c
            # diff = upper - lower
            # ax.axvline(lower, 0, 1, ls="--", color=c, label="68 % [{:.3}]".format(diff))
            # ax.axvline(mid, 0, 1, ls="-", color=c, label="mean [{:.3}]".format(mid))
            # ax.axvline(upper, 0, 1, ls="--", color=c)

        if params_print == None:
            ax.set_xlabel(r"$%s_{reco} - %s_{true}$" % (p,p))
        else:
            p_print = params_print[p]
            if p == "energy":
                ax.set_xlabel(r"$\frac{%s_{reco} - %s_{true}}{%s_{true}}$ [%%]" % (p_print,p_print,p_print))
            else:
                ax.set_xlabel(r"$%s_{reco} - %s_{true}$" % (p_print,p_print))
        ax.set_ylabel("N-events")
        ax.legend()
        plt.tight_layout()
        plt.savefig( "plots/1d_hist/%s_%s" % (pdf_name, p), dpi=300)
        plt.close()
        # plt.savefig( "plots/FinalPlotsDeepCoreNue/1d_hist/%s_%s" % (pdf_name, p), dpi=300)
        # pdf.savefig()
    # pdf.close()

def plot_2dhist(fits, labels, plot_dict, labels_print=None, num_bins=8, figname="2dhist", Erange=None):
    assert len(labels) == 2

    folder = "plots/2d_hist/"+"/".join(figname.split("/")[:-1])
    if not os.path.exists(folder):
        os.makedirs(folder)


    params_print = dict(x="x", y="y", z="z", time="t", zenith=r"\theta", azimuth="azi", length="len", energy="E")
    params = ["x", "y", "z", "time", "azimuth", "zenith", "energy"]
    fit1 = fits[labels[0]]
    fit2 = fits[labels[1]]
    true = fits["true"]
    x = fits["true"]["energy"]

    if labels_print:
        labels = labels_print

    for p in params:
        if p == "energy": 
            y1 = (fit1[p] - true[p])/true[p]*100
            y2 = (fit2[p] - true[p])/true[p]*100
        else:
            y1 = fit1[p] - true[p]
            y2 = fit2[p] - true[p]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24,9), sharey=False)
        if Erange is None:
            xbins = get_bins(1., 100., num=num_bins)
        else:
            xbins = get_bins(Erange[0], Erange[1], num=num_bins)
    #     xbins = np.logspace(1.2, 2, num_bins+1)
        ybins = get_bins(plot_dict[p][0], plot_dict[p][1],num=num_bins)

        for y, ax, label in zip([y1, y2], [ax1, ax2], labels):
            hist = Histogram( 2, bins=[xbins,ybins], x=x, y=y)

            # Convert to mHz
            hist *= 1.e3

            # Plot hist
            cmesh = plot_hist( ax=ax, hist=hist, text=False, edges=False, cmap="Oranges" )

            # Plot the y percentiles
            plot_y_percentiles(
                ax=ax,
                x_bin_centers=hist.bin_centers("x"),
                x_bin_edges=hist.bin_edges("x"),
                x=x,
                y=y,
            )

            pp = params_print[p]
            # if p == "energy":
            #     ax.set_ylabel(r"$\frac{%s_{reco} - %s_{true}}{%s_{true}}$" % (pp, pp, pp), size=35 )
            # else:
            #     ax.set_ylabel(r"$%s_{reco} - %s_{true}$" % (pp, pp) )
            ax.set_ylim(plot_dict[p][0], plot_dict[p][1])
            ax.legend([r"1 $\sigma$", "median"], prop={"size":20})
            ax.set_title(r"%s, $\nu_{e}$" % (label), size=30)
            ax.set_xlabel(r"$E_{\nu ,true}$ [GeV]", size=35)
        if p == "energy":
            # ax1.set_ylabel(r"$\frac{%s_{reco} - %s_{true}}{%s_{true}}$" % (pp, pp, pp), size=35 )
            ax1.set_ylabel(r"$(%s_{reco} - %s_{true}) / %s_{true}$ [%%]" % (pp, pp, pp), size=35 )
        elif p in ["zenith", "azimuth"]:
            ax1.set_ylabel(r"$%s_{reco} - %s_{true}$ [rad]" % (pp, pp), size=35 )
        else:
            ax1.set_ylabel(r"$%s_{reco} - %s_{true}$" % (pp, pp), size=35 )
        plt.tight_layout()

        plt.savefig( "plots/2d_hist/%s_%s" % (figname, p), dpi=300)
        plt.close()
        # plt.savefig( "plots/FinalPlotsDeepCoreNue/2d_hist/%s_%s_%s" % (labels[0], labels[1], p), dpi=300)


def plot_2dhist_energy(data_Efit, num_bins=8, figname="2dhist_Efit", Erange=None):

    fig, ax = plt.subplots(figsize=(24,9))

    remove_failed = True
    failed = 0

    diffs = []
    Etrues = []
    for i in range(len(data_Efit)):
        if data_Efit[i]["InfoGeneral"]["fit"].values[0]:
            Efit = data_Efit[i]["MonopodFitFinal"].energy.values
        else:
            Efit = data_Efit[i]["MonopodFitFinalMinimizerSteps"].energy.values

        if remove_failed:
            mask = data_Efit[i]["MonopodFitFinal"]["fit_status"] == 0
            failed += sum(~mask)
            Etrue = data_Efit[i]["MCNeutrino"].energy.values[mask]
            Efit = Efit[mask]
            Etrues.append(Etrue)
            diff = (Efit-Etrue)/Etrue      
        else:
            Etrue = data_Efit[i]["MCNeutrino"].energy.values
            Etrues.append(Etrue)
            diff = (Efit-Etrue)/Etrue
        # print len(diff)
        diffs.append(diff)
    diffs = np.concatenate(diffs)
    Etrues = np.concatenate(Etrues)
    print len(diffs), "fits included"
    if remove_failed:
        print failed, "failed fits removed"
    x = Etrues
    y = diffs

    if Erange is None:
        xbins = get_bins(1., 100., num=num_bins)
    else:
        xbins = get_bins(Erange[0], Erange[1], num=num_bins)
    ybins = get_bins(-1, 1, num=num_bins)

    hist = Histogram( 2, bins=[xbins,ybins], x=x, y=y)

    # Convert to mHz
    hist *= 1.e3

    # Plot hist
    cmesh = plot_hist( ax=ax, hist=hist, text=False, edges=False, cmap="Oranges" )

    # Plot the y percentiles
    plot_y_percentiles(
        ax=ax,
        x_bin_centers=hist.bin_centers("x"),
        x_bin_edges=hist.bin_edges("x"),
        x=x,
        y=y,
    )

    ax.set_ylim(-1, 1)
    ax.legend([r"1 $\sigma$", "median"], prop={"size":20})
    ax.set_title(r"%s, $\nu_{e}$" % (figname), size=30)
    ax.set_xlabel(r"$E_{\nu ,true}$ [GeV]", size=35)
    ax.set_ylabel(r"$(E_{reco} - E_{true}) / E_{true}$" , size=35 )
    ax.axhline(0, 0, 1, color="k", lw=2)

    plt.savefig( "plots/2d_hist/Efit/%s" % (figname), dpi=300)
    # plt.savefig( "plots/2d_hist/os100/%s" % (figname), dpi=300)
        # plt.savefig( "plots/FinalPlotsDeepCoreNue/2d_hist/%s_%s_%s" % (labels[0], labels[1], p), dpi=300)


def check_dir(path):
    import os
    folder = "/".join(path.split("/")[:-1])
    if not os.path.exists(folder):
        os.makedirs(folder)

def plot_minimizer(data, k, seed="Monopod_best", llh_truth=None, string="minimizerplot", energy_only=False, MAP="plasma", showplot=True): #, llhs):
    
    events = data["MCNeutrino"].Event.unique()
    event = events[k]
    if energy_only:
        params = ["E"]
    else:
        params = ["x", "y", "z", "t", "azi", "zen", "E"]
    
    oversampling = data["InfoGeneral"]["Oversampling"].values[0]

    figfolder = "/home/thomas/Documents/master_thesis/DirectReco/ICU/figures/MinizerPlots/"

    if not energy_only:
        figname = figfolder + '/{}_{}_{}.pdf'.format(event, string, oversampling)
        pdf = PdfPages(figname)
    else:
        figname = figfolder + '/{}_{}_{}.png'.format(event, string, oversampling)

    check_dir(figname)
    
    if llh_truth:
        llhs = true["MonopodFitFinal"]
        Tllh = llhs[llhs.Event == event].logl.values[0]

    minSteps = data["MonopodFitFinalMinimizerSteps"]
    minSteps = minSteps[minSteps.Event == event]

    stepDict = dict(x = minSteps.x, y = minSteps.y, z = minSteps.z,
                    t = minSteps.time, zen = minSteps.zenith, azi = minSteps.azimuth,
                    E = minSteps.energy, l=minSteps.length, llh = minSteps.speed)
    
    truth = data["MCNeutrino"]
    truth = truth[truth.Event == event]

    trueDict = dict(x = truth.x, y = truth.y, z = truth.z,t = truth.time, 
                    zen = truth.zenith, azi = truth.azimuth, E = truth.energy)    
    
    best = data[seed]
    best = best[best.Event == event]

    bestDict = dict(x = best.x, y = best.y, z = best.z,t = best.time, 
                    zen = best.zenith, azi = best.azimuth, E = best.energy)

    for p in params:
        x = stepDict[p]
        xt = trueDict[p]
        xb = bestDict[p]
        y = np.arange(len(x))
        cm = plt.get_cmap(MAP)

        llh = stepDict["llh"].values
        norm = mpl.colors.Normalize(vmin=min(llh), vmax=max(llh))
        llh_n = norm(llh)

        NPOINTS = len(x)
        fig = plt.figure(figsize=(14,8))
        N = 12
        # ax1 = plt.subplot2grid((N,N), (0,0), colspan=N, rowspan=N-2)
        # ax2 = plt.subplot2grid((N,N), (N-2,0), colspan=N)

        ax1 = plt.subplot2grid((N,N), (0,0), colspan=N-3, rowspan=N-2)
        ax2 = plt.subplot2grid((N,N), (N-1,0), colspan=N-3, rowspan=N)

        colLabels = ["Parameter", "Value"]
        info = data["InfoGeneral"].copy()
        info["RunTime"] = np.round(info["RunTime"]/3.6e6, 2)
        infoLabels = ["Oversampling", "RunTime", "DIMA", "MeanSPE", "PhotonsPerBin"]
        infoLabelDict = {"Oversampling":"OS", "RunTime":"Time[Hrs]", "DIMA":"DIMA", "MeanSPE":"SPE", "PhotonsPerBin":"PPB"}
        cellText = [(infoLabelDict[l], info[l].values[0]) for l in infoLabels]

        cellTrue = xt.values.round(2)[0]
        cellFit  = x.iloc[-1].round(2)
        cellSeed = xb.values.round(2)[0]
        cellText += [(r"$%s_{true}$"%p, cellTrue), (r"$%s_{fit}$"%p, cellFit), (r"$%s_{seed}$"%p, cellSeed)]
        cellText += [(r"$\Delta %s_{fit}$"%p, np.round(cellTrue-cellFit, 2)), (r"$\Delta %s_{seed}$"%p, np.round(cellTrue-cellSeed, 2))]

        tb = ax1.table(cellText=cellText, colWidths=[.4,.3], 
               colLabels=colLabels, bbox=[1.05, .4, .25, .62])

        prop = tb.properties()
        cells = prop["child_artists"]
        for c in cells:
            c.set_linewidth(2)
            c._loc = "left"
            c.set_fontsize(15)


        color=cm(llh_n)
        scatter = ax1.scatter(y, x, alpha=0.9, color=color, label="Minimizer Step")
        ax1.hlines(xt, 0, NPOINTS, label="True %s" % (p))
        ax1.hlines(xb, 0, NPOINTS, label="seed %s" % (p), color="green", lw=1)
        ax1.set_ylabel(p, size=25)
        ax1.set_xlabel("Iteration", size=15)
        if llh_truth:
            ax1.set_title(r"$llh_{true}$: %4.6s, $llh_{fit}$: %4.6s, event: %s" %(Tllh, -llh[-1], event), size=20)
        else:
            ax1.set_title(r"$llh_{fit}$: %4.6s, event: %s" %(-llh[-1], event), size=20)
        cb1 = mpl.colorbar.ColorbarBase(ax2, cmap=MAP,
                                        norm=norm,
                                        orientation='horizontal')
        cb1.set_label('llh', size=15)

        labels = ["{}".format(int(l)) for l in llh]

        ax1.legend()
        plt.tight_layout()
        if energy_only:
            plt.savefig( figname, dpi=300)
        else:
            pdf.savefig()
        if not showplot:
            plt.close()
    if not energy_only:
        pdf.close()

def plot_llh_std(data, oss, label):
    plt.figure()
    for dat, ovs in zip(data, oss):
        llhs = []
        for d in dat:
            llh = -d["MonopodFitFinalMinimizerSteps"]["speed"].values[0]
            llhs.append(llh)
        llhs = np.array(llhs)

        plt.scatter(ovs, llhs.std())
    plt.xscale("log")
    plt.xlabel("Oversampling")
    plt.ylabel(r"$\sigma$")
    plt.title("Seed: " + label)


def plot_llh_points(data, oss, label):
    plt.figure()
    for dat, ovs in zip(data, oss):
        llhs = []
        for d in dat:
            llh = -d["MonopodFitFinalMinimizerSteps"]["speed"].values[0]
            llhs.append(llh)
        llhs = np.array(llhs)
        ovs_plot = [ovs for i in range(len(llhs))]
        plt.scatter(ovs_plot, llhs)
    plt.xscale("log")
    plt.xlabel("Oversampling")
    plt.ylabel("llh")
    plt.title("Seed: " + label)


def plot_llh_points_single(data, oss, k, label):
    llhs = []
    o = oss[k]
    for d in data[k]:
        llh = -d["MonopodFitFinalMinimizerSteps"]["speed"].values[0]
        llhs.append(llh)
    llhs = np.array(llhs)
    ovs = [o for i in range(len(llhs))]
    plt.figure()
    plt.scatter(ovs, llhs)
    plt.xlabel("Oversampling")
    plt.ylabel("llh")
    plt.title("Seed: " + label)

def plot_LLH_scan(data, event, pair, fit, N=30, vmax=30, figfolder="/home/thomas/Documents/master_thesis/DirectReco/ICU/figures/LLH_scans/", 
                  figname="LLH_scan", draw_minimizer=False, draw_numbers=False):
    from scipy import stats


    oversampling = data[event][pair][0]["InfoGeneral"]["Oversampling"].values[0]

    # Get the 2 scan parameters
    p1 = pair.split("_")[0]
    p2 = pair.split("_")[1]

    llhs = []
    p1s = []
    p2s = []
    dat = data[event][pair]
    # Extract parameters from files
    for i in range(len(dat)):
        p = dat[i]["MonopodFitFinalMinimizerSteps"]
        llhs.append(-p.speed)
        p1s.append(p[p1])
        p2s.append(p[p2])


    # Get true parameters
    # xtrue = dat[i]["MCNeutrino"][p1].values[0]
    # ytrue = dat[i]["MCNeutrino"][p2].values[0]
    xtrue = dat[i]["I3MCTree"].iloc[0][p1]#.values[0]
    ytrue = dat[i]["I3MCTree"].iloc[0][p2]#.values[0]
    
    # Rearange into grids
    p1s = np.array(p1s)
    p2s = np.array(p2s)
    print p1, p1s.min(), p1s.max(), p2, p2s.min(), p2s.max()
    
    llhs = np.array(llhs)
    llhs.resize((N, N))

    p1s.resize((N, N))
    p1s_contour = p1s.copy()
    p1sdiff = p1s[1,0] - p1s[0,0]
    p1s = np.append(p1s[:,0], p1s[-1,0]+p1sdiff)
    p1s = np.array([p1s for i in range(N+1)]).T

    p2s.resize((N, N))
    p2s_contour = p2s.copy()
    p2sdiff = p2s[0,1] - p2s[0,0]
    p2s = np.append(p2s[0,:], p2s[0,-1]+p2sdiff)
    p2s = np.array([p2s for i in range(N+1)])

    # Calculate precise sigma percentage 
    sig1 = 1-stats.norm.cdf(-1)*2
    sig2 = 1-stats.norm.cdf(-2)*2
    sig3 = 1-stats.norm.cdf(-3)*2
    # Find the contour levels for degrees of freedom = df
    df = 7
    l1, l2, l3 = stats.chi2.ppf(sig1, df), stats.chi2.ppf(sig2, df), stats.chi2.ppf(sig3, df)

    # Calculate the Log-likelihood ratio for the grid
    LLR = 2*(llhs - llhs.min())
    LLR_min = LLR.min()
    levels = [LLR_min + l1, LLR_min + l2, LLR_min + l3]
    print levels

    # Find the best point in the grid
    xi, yi = np.where(LLR == LLR.min())
    bestx, besty = p1s[xi, yi], p2s[xi, yi]

    # Extract parameters from the best fit
    minSteps = fit["MonopodFitFinalMinimizerSteps"]
    x = minSteps[p1].values
    y = minSteps[p2].values
    llhs_fit = -minSteps["speed"].values
    llh_seed = llhs_fit[0]
    llh_best_fit = llhs_fit[-1]
    event = minSteps.Event.values[0]


    # Estimate LLH at truth
    # print p1s.shape, p2s.shape, xtrue, ytrue
    mat = np.sqrt((p1s - xtrue)**2 + (p2s - ytrue)**2)
    index = np.where(mat == mat.min())
    LLHtrue = llhs[index[0], index[1]][0]

    print "LLH Best-Fit: {}, LLH Scan: {}".format(llh_best_fit, llhs.min())

    fig = plt.figure(figsize=(16,9))
    ax = plt.axes()
    plt.contour(p1s_contour, p2s_contour, LLR, levels=levels, colors=["purple", "blue", "green"])
    plt.pcolormesh(p1s-p1sdiff/2., p2s-p2sdiff/2., LLR, cmap="hot_r", vmax=vmax)

    if draw_numbers:
        for (i, j), z in np.ndenumerate(llhs):
            # print p1s[i, j], p2s[i, j], z
            ax.text(p1s[i, j], p2s[i, j], '{:0.1f}'.format(z), ha='center', va='center', color="c", fontsize=10)

    cb = plt.colorbar()
    plt.xlabel(p1)
    plt.ylabel(p2)
    cb.ax.set_ylabel(r"2$\Delta$LLH")

    if draw_minimizer:
        bestxs = [x[0]]
        bestys = [y[0]]
        bestLLHs = [llhs_fit[0]]
        for l, x_, y_ in zip(llhs_fit, x, y):
            if l < bestLLHs[-1]:
                bestLLHs.append(l)
                bestxs.append(x_)
                bestys.append(y_)
        cmap1 = mpl.cm.get_cmap("cool_r") #cool_r
        cmap2 = mpl.cm.get_cmap("plasma")
        colors_llh = cmap1(np.linspace(0, 1, len(bestxs)))
        colors_move = cmap2(np.linspace(0, 1, len(bestxs)))
        plt.scatter(bestxs, bestys, c=colors_llh, zorder=3)
        for i in range(len(bestxs)):
            plt.plot(bestxs[i:i+2], bestys[i:i+2], color=colors_move[i], lw=2)

        # plt.scatter(x, y, c=colors, label="Iterations")
        # plt.plot(x, y, c="c", lw=1)
    plt.scatter(x[0], y[0], marker="x", s=100, lw=5, color="g", zorder=3, label="Seed [%.5s]" % (llh_seed))
    plt.scatter(x[-1], y[-1], marker="x", s=100, lw=5, color="b", zorder=3, label="BestFit [%.5s]" % (llh_best_fit))
    plt.scatter(xtrue, ytrue, marker="x", s=100, lw=5, color="purple", zorder=3, label="Truth [~%.5s]" % (LLHtrue))
    plt.scatter(bestx, besty, marker="x", s=100, lw=5, color="k", zorder=3, label="MinLLHScan [%.5s]" % (llhs.min()))

    # plt.xlim(p1s_contour.min(), p1s_contour.max())
    # plt.ylim(p2s_contour.min(), p2s_contour.max())

    plt.title(figname + ", event: {}, {}".format(event, oversampling))
    plt.legend()
    plt.tight_layout()
    plt.savefig(figfolder + "{}_{}_{}_{}_{}".format(figname, event, p1, p2, oversampling))

def plot_LLH_scan_mergedhdf5(data, event, pair, fit=None, N=15, vmax=30, figfolder="/home/thomas/Documents/master_thesis/DirectReco/ICU/figures/LLH_scans/", 
                  figname="LLH_scan", draw_minimizer=False, draw_numbers=False):
    from scipy import stats

    data = data[event]
    oversampling = data["InfoGeneral"]["Oversampling"]
    # Get the 2 scan parameters
    p1 = pair.split("_")[0]
    p2 = pair.split("_")[1]

    llhs = []
    # Extract parameters from files
    p1s = data["MonopodFitFinalMinimizerSteps"][p1].values
    p2s = data["MonopodFitFinalMinimizerSteps"][p2].values
    llhs = -data["MonopodFitFinalMinimizerSteps"]["speed"]
    
    # Get true parameters
    xtrue = data["I3MCTree"][p1]
    ytrue = data["I3MCTree"][p2]
    
    # Rearange into grids
    p1s = np.array(p1s)
    p2s = np.array(p2s)
    print p1, p1s.min(), p1s.max(), p2, p2s.min(), p2s.max()
    
    llhs = np.array(llhs)
    llhs.resize((N, N))

    p1s.resize((N, N))
    p1s_contour = p1s.copy()
    p1sdiff = p1s[1,0] - p1s[0,0]
    p1s = np.append(p1s[:,0], p1s[-1,0]+p1sdiff)
    p1s = np.array([p1s for i in range(N+1)]).T

    p2s.resize((N, N))
    p2s_contour = p2s.copy()
    p2sdiff = p2s[0,1] - p2s[0,0]
    p2s = np.append(p2s[0,:], p2s[0,-1]+p2sdiff)
    p2s = np.array([p2s for i in range(N+1)])

    # Calculate precise sigma percentage 
    sig1 = 1-stats.norm.cdf(-1)*2
    sig2 = 1-stats.norm.cdf(-2)*2
    sig3 = 1-stats.norm.cdf(-3)*2
    # Find the contour levels for degrees of freedom = df
    df = 7
    l1, l2, l3 = stats.chi2.ppf(sig1, df), stats.chi2.ppf(sig2, df), stats.chi2.ppf(sig3, df)

    # Calculate the Log-likelihood ratio for the grid
    LLR = 2*(llhs - llhs.min())
    LLR_min = LLR.min()
    levels = [LLR_min + l1, LLR_min + l2, LLR_min + l3]
    print levels

    # Find the best point in the grid
    xi, yi = np.where(LLR == LLR.min())
    bestx, besty = p1s[xi, yi], p2s[xi, yi]

    if fit is not None:
        # Extract parameters from the best fit
        minSteps = fit["MonopodFitFinalMinimizerSteps"]
        x = minSteps[p1].values
        y = minSteps[p2].values
        llhs_fit = -minSteps["speed"].values
        llh_seed = llhs_fit[0]
        llh_best_fit = llhs_fit[-1]
        event = minSteps.Event.values[0]
        print "LLH Best-Fit: {}, LLH Scan: {}".format(llh_best_fit, llhs.min())

    # Estimate LLH at truth
    # print p1s.shape, p2s.shape, xtrue, ytrue
    mat = np.sqrt((p1s - xtrue)**2 + (p2s - ytrue)**2)
    index = np.where(mat == mat.min())
    LLHtrue = llhs[index[0], index[1]][0]

    fig = plt.figure(figsize=(16,9))
    ax = plt.axes()
    plt.contour(p1s_contour, p2s_contour, LLR, levels=levels, colors=["purple", "blue", "green"])
    plt.pcolormesh(p1s-p1sdiff/2., p2s-p2sdiff/2., LLR, cmap="hot_r", vmax=vmax)

    if draw_numbers:
        for (i, j), z in np.ndenumerate(llhs):
            # print p1s[i, j], p2s[i, j], z
            ax.text(p1s[i, j], p2s[i, j], '{:0.1f}'.format(z), ha='center', va='center', color="c", fontsize=10)

    cb = plt.colorbar()
    plt.xlabel(p1)
    plt.ylabel(p2)
    cb.ax.set_ylabel(r"2$\Delta$LLH")

    if draw_minimizer:
        bestxs = [x[0]]
        bestys = [y[0]]
        bestLLHs = [llhs_fit[0]]
        for l, x_, y_ in zip(llhs_fit, x, y):
            if l < bestLLHs[-1]:
                bestLLHs.append(l)
                bestxs.append(x_)
                bestys.append(y_)
        cmap1 = mpl.cm.get_cmap("cool_r") #cool_r
        cmap2 = mpl.cm.get_cmap("plasma")
        colors_llh = cmap1(np.linspace(0, 1, len(bestxs)))
        colors_move = cmap2(np.linspace(0, 1, len(bestxs)))
        plt.scatter(bestxs, bestys, c=colors_llh, zorder=3)
        for i in range(len(bestxs)):
            plt.plot(bestxs[i:i+2], bestys[i:i+2], color=colors_move[i], lw=2)

    if fit is not None:
        plt.scatter(x[0], y[0], marker="x", s=100, lw=5, color="g", zorder=3, label="Seed [%.5s]" % (llh_seed))
        plt.scatter(x[-1], y[-1], marker="x", s=100, lw=5, color="b", zorder=3, label="BestFit [%.5s]" % (llh_best_fit))
    plt.scatter(xtrue, ytrue, marker="x", s=100, lw=5, color="purple", zorder=3, label="Truth [~%.5s], E [%.5s]" % (LLHtrue, ytrue))
    plt.scatter(bestx, besty, marker="x", s=100, lw=5, color="k", zorder=3, label="MinLLHScan [%.5s], E [%.5s]" % (llhs.min(), besty[0]))

    print "Etrue/Ebest: ", ytrue/besty[0]
    print bestx, besty

    plt.title(figname + ", event: {}, {}".format(event, oversampling))
    plt.legend()
    plt.tight_layout()
    plt.savefig(figfolder + "{}_{}_{}_{}_{}".format(figname, event, p1, p2, oversampling))

def plot_LLH_scan_llhrepeat(data, event, pair, N=30, vmax=30, figname="LLH_scan", draw_minimizer=False, draw_numbers=False):
    from scipy import stats
    # Get the 2 scan parameters
    p1 = pair.split("_")[0]
    p2 = pair.split("_")[1]

    llhs = []
    p1s = []
    p2s = []
    dat = data[event][pair]
    # Extract parameters from files
    for i in range(len(dat)):
        d = dat[i]
        p = d["FixedSeed"]
        p1s.append(p[p1])
        p2s.append(p[p2])

        monofitfinal_keys = [k for k in d.keys() if "MonopodFitFinal" in k and "Minimizer" not in k]
        llh_temp = []
        for k in monofitfinal_keys:
            llh_temp.append(d[k].logl.values[0])
        llhs.append(np.mean(llh_temp))

    # Get true parameters
    xtrue = dat[i]["MCNeutrino"][p1].values[0]
    ytrue = dat[i]["MCNeutrino"][p2].values[0]
    event = dat[i]["MCNeutrino"].Event.values[0]
    # Rearange into grids
    p1s = np.array(p1s)
    p2s = np.array(p2s)
    llhs = np.array(llhs)

    llhs.resize((N, N))

    p1s.resize((N, N))
    p1s_contour = p1s.copy()
    p1sdiff = p1s[1,0] - p1s[0,0]
    p1s = np.append(p1s[:,0], p1s[-1,0]+p1sdiff)
    p1s = np.array([p1s for i in range(N+1)]).T

    p2s.resize((N, N))
    p2s_contour = p2s.copy()
    p2sdiff = p2s[0,1] - p2s[0,0]
    p2s = np.append(p2s[0,:], p2s[0,-1]+p2sdiff)
    p2s = np.array([p2s for i in range(N+1)])

    # Calculate precise sigma percentage 
    sig1 = 1-stats.norm.cdf(-1)*2
    sig2 = 1-stats.norm.cdf(-2)*2
    sig3 = 1-stats.norm.cdf(-3)*2
    # Find the contour levels for degrees of freedom = df
    df = 7
    l1, l2, l3 = stats.chi2.ppf(sig1, df), stats.chi2.ppf(sig2, df), stats.chi2.ppf(sig3, df)

    # Calculate the Log-likelihood ratio for the grid
    LLR = 2*(llhs - llhs.min())
    LLR_min = LLR.min()
    levels = [LLR_min + l1, LLR_min + l2, LLR_min + l3]
    print levels

    # Find the best point in the grid
    xi, yi = np.where(LLR == LLR.min())
    bestx, besty = p1s[xi, yi], p2s[xi, yi]

    # Extract parameters from the best fit
    # minSteps = fit["MonopodFitFinalMinimizerSteps"]
    # x = minSteps[p1].values
    # y = minSteps[p2].values
    # llhs_fit = -minSteps["speed"].values
    # llh_seed = llhs_fit[0]
    # llh_best_fit = llhs_fit[-1]
    # event = minSteps.Event.values[0]

    # Estimate LLH at truth
    # print p1s.shape, p2s.shape, xtrue, ytrue
    mat = np.sqrt((p1s - xtrue)**2 + (p2s - ytrue)**2)
    index = np.where(mat == mat.min())
    LLHtrue = llhs[index[0], index[1]][0]

    # print "LLH Best-Fit: {}, LLH Scan: {}".format(llh_best_fit, llhs.min())

    fig = plt.figure(figsize=(16,9))
    ax = plt.axes()
    plt.contour(p1s_contour, p2s_contour, LLR, levels=levels, colors=["purple", "blue", "green"])
    plt.pcolormesh(p1s-p1sdiff/2., p2s-p2sdiff/2., LLR, cmap="hot_r", vmax=vmax)

    if draw_numbers:
        for (i, j), z in np.ndenumerate(llhs):
            # print p1s[i, j], p2s[i, j], z
            ax.text(p1s[i, j], p2s[i, j], '{:0.1f}'.format(z), ha='center', va='center', color="c", fontsize=10)

    cb = plt.colorbar()
    plt.xlabel(p1)
    plt.ylabel(p2)
    cb.ax.set_ylabel(r"2$\Delta$LLH")

    if draw_minimizer:
        bestxs = [x[0]]
        bestys = [y[0]]
        bestLLHs = [llhs_fit[0]]
        for l, x_, y_ in zip(llhs_fit, x, y):
            if l < bestLLHs[-1]:
                bestLLHs.append(l)
                bestxs.append(x_)
                bestys.append(y_)
        cmap1 = mpl.cm.get_cmap("cool_r") #cool_r
        cmap2 = mpl.cm.get_cmap("plasma")
        colors_llh = cmap1(np.linspace(0, 1, len(bestxs)))
        colors_move = cmap2(np.linspace(0, 1, len(bestxs)))
        plt.scatter(bestxs, bestys, c=colors_llh, zorder=3)
        for i in range(len(bestxs)):
            plt.plot(bestxs[i:i+2], bestys[i:i+2], color=colors_move[i], lw=2)

        # plt.scatter(x, y, c=colors, label="Iterations")
        # plt.plot(x, y, c="c", lw=1)
    # plt.scatter(x[0], y[0], marker="x", s=100, lw=5, color="g", zorder=3, label="Seed [%.5s]" % (llh_seed))
    # plt.scatter(x[-1], y[-1], marker="x", s=100, lw=5, color="b", zorder=3, label="BestFit [%.5s]" % (llh_best_fit))
    plt.scatter(xtrue, ytrue, marker="x", s=100, lw=5, color="purple", zorder=3, label="Truth [~%.5s]" % (LLHtrue))
    plt.scatter(bestx, besty, marker="x", s=100, lw=5, color="k", zorder=3, label="MinLLHScan [%.5s]" % (llhs.min()))

    # plt.xlim(p1s_contour.min(), p1s_contour.max())
    # plt.ylim(p2s_contour.min(), p2s_contour.max())
    plt.title(figname + ", event: {}".format(event))
    plt.legend()
    plt.tight_layout()
    plt.savefig("plots/LLH_scans/{}_{}_{}_{}".format(figname, event, p1, p2))



def make_fig_3D(figsize=(8,8)):

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    return ax

