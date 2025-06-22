import numpy as np
import os
import os.path as osp
import pandas as pd
import math
import sys
from itertools import chain
from mne.filter import notch_filter, filter_data, resample
from keras.models import model_from_json
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score
from imblearn.metrics import specificity_score
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.backends.backend_pdf as backend_pdf

# seed_range = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
# seed_range = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
seed_range = range(1, 51)

prepare = False  # True: generate data; False: combine results
contain_spikefile_in_bckg = True  # default: False (only crop normal files); True: include spikefiles in background
draw = False  # default: False, True if we need viz to check signals
random = True
SSDname = 'SSD_spikenet_bth_EQ'  # default: 'SSD_spikenet_btheeg' (same ratio as MEG), 'SSD_spikenet_btheeg_LS' for more background samples

# # settings
# np.random.seed(1)

# path = osp.join('C:', osp.sep, 'Users', 'FBE', 'Dropbox', 'Data', 'Spike')
path = osp.join(osp.sep, 'Users', 'shirleywei', 'Dropbox', 'Data', 'Spike')
files = os.listdir(osp.join(path, 'newEEGdata'))
filesubs = [f[:-4] for f in files]

# global var
notch_freq = 50  # for eeg data from China
bp_freq = [0.5, None]
org_channels = ['FP1', 'FP2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'FZ',
                'CZ', 'PZ']
mono_channels = ['FP1', 'F3', 'C3', 'P3', 'F7', 'T3', 'T5', 'O1', 'FZ', 'CZ', 'PZ', 'FP2', 'F4', 'C4', 'P4', 'F8', 'T4',
                 'T6', 'O2']
bipolar_channels = ['FP1-F7', 'F7-T3', 'T3-T5', 'T5-O1', 'FP2-F8', 'F8-T4', 'T4-T6', 'T6-O2', 'FP1-F3', 'F3-C3',
                    'C3-P3', 'P3-O1', 'FP2-F4', 'F4-C4', 'C4-P4', 'P4-O2', 'FZ-CZ', 'CZ-PZ']
Fs = 128
L = int(round(1 * Fs))
step = 1
batch_size = 256


def read_m00_file(path):
    # read data
    f = open(path, "r")
    txt = f.readlines()[0]
    data = np.loadtxt(path, skiprows=2)
    sfreq = 1000 / float(txt.split(' ')[3].split('=')[1])
    downrate = Fs

    seg = data[:, :19].transpose()
    data = np.where(np.isnan(seg), 0, seg)
    data = - data

    # switch rows
    switch_idx = [org_channels.index(mono_channels[i]) for i in range(19)]
    data = data[switch_idx, :]

    # montages
    bipolar_ids = np.array(
        [[mono_channels.index(bc.split('-')[0]), mono_channels.index(bc.split('-')[1])] for bc in bipolar_channels])
    bipolar_data = data[bipolar_ids[:, 0]] - data[bipolar_ids[:, 1]]
    average_data = data - data.mean(axis=0)
    res = np.concatenate([average_data, bipolar_data], axis=0)

    # downsample
    res = resample(res, up=1, down=sfreq / downrate)

    return res


def preprocess_eeg(X):
    # Notch and highpass
    X = filter_data(X, Fs, bp_freq[0], bp_freq[1], n_jobs=-1, method='fir', verbose=False)
    X = notch_filter(X, Fs, notch_freq, n_jobs=-1, method='fir', verbose=False)

    return X


# parameters
downrate = Fs
nX = 128
nZ = 0
if (SSDname == 'SSD_spikenet_btheeg') or (SSDname == 'SSD_spikenet_bth'):
    ratio = 0.37455223679603544  # 1/0 ratio according to MEG data
elif SSDname[-2:] == 'LS':
    ratio = 0.1
else:
    ratio = 0.5  # do not use this

# -----------------
nT = nX + nZ * 2
nt = int(nT / 2)  # half length

# read seizure and spike file names
# seizure
seifile = pd.read_excel(osp.join(path, 'newEEGdata_tb', 'seionset.xlsx'))
seifile = seifile.dropna()
seifilesub = seifile['file'].unique()
# spikes
spikefile = pd.read_excel(osp.join(path, 'newEEGdata_tb', 'feirevised_filter.xlsx'), header=1)
interictal_filesub = spikefile['interictal_file'].dropna().unique()
ictal_filesub = spikefile['ictal_file'].dropna().unique()
abfilesub = np.array(list(set(interictal_filesub.tolist() + ictal_filesub.tolist() + seifilesub.tolist())))
normalfilesub = [f for f in filesubs if f not in abfilesub]
# background
bckgfile = pd.read_excel(osp.join(path, 'newEEGdata_tb', 'normal_files.xlsx'), header=0)
bckgfilesub = bckgfile['file'].dropna().unique()

# create spike file dict
filedict = dict()  # 'filesub': [time1, time2, ...] (seconds) if time is empty then normal subs
spikefiledict = dict()
normalfiledict = dict()
bckgfiledict = dict()
spikefile = spikefile[spikefile['comments'].isna()]  # remove red rows (not sure)
spikefilesub = spikefile['interictal_file'].dropna().unique().tolist()
numspikefile = 0
for f in spikefilesub:
    df = spikefile[spikefile['interictal_file'] == f]
    extime = df['exact_time'].unique().tolist()
    sec = [t.minute * 60 + t.second + t.microsecond / 1e6 for t in extime]  # convert to seconds
    filedict[f] = sec
    spikefiledict[f] = sec
    numspikefile += len(sec)
for f in normalfilesub:
    filedict[f] = list()
    normalfiledict[f] = list()
for f in bckgfilesub:
    df = bckgfile[bckgfile['file'] == f]
    sec = df['time'].unique().tolist()
    bckgfiledict[f] = sec

# spike files and normal ones
# randn = math.ceil((numspikefile * ((1 - ratio) / ratio)) / len(normalfilesub))

# ==============================================

# I/O directories
targetDir = "Output/"
dataDir = "/Users/shirleywei/Dropbox/Projects/NestedDeepLearningModel/Dataset/"
if not os.path.exists(dataDir + SSDname):
    os.makedirs(dataDir + SSDname)

# load model
with open("model/spikenet1.o_structure.txt", "r") as ff:
    json_string = ff.read()

model = model_from_json(json_string)
model.load_weights("model/spikenet1.o_weights.h5")

oos = {'recall': [],
       'prec': [],
       'spec': [],
       'f1': [],
       'prauc': [],
       'auc': []}

REC, PREC, SPEC, F1, PRAUC, AUC = [], [], [], [], [], []
index_col = []

for seed in seed_range:

    if prepare:
        np.random.seed(seed)

        # initialization
        y, yp = [], []
        X = np.array([]).reshape(0, 128, 37)
        sampleID = 0
        samplespike = 0
        samplenormal = 0

        original_stdout = sys.stdout

        with open(dataDir + SSDname + '/seed' + str(seed) + '.txt', 'w') as f:
            sys.stdout = f

            # =============WRITE START=================================
            for filesub in spikefiledict.keys():
                filename = osp.join(path, 'newEEGdata', filesub + '.m00')
                if osp.exists(filename):
                    # read X
                    eeg = read_m00_file(filename)
                    eeg = preprocess_eeg(eeg)

                    # ---------------------------------------
                    # read y
                    times = np.array(spikefiledict[filesub])
                    msm = eeg.shape[1]  # total number of measurements
                    if len(times) == 0:
                        continue
                    else:
                        pos = np.round(times * downrate)
                        for p in pos:
                            p = round(p)
                            subeeg = eeg[:, (p - nt): (p + nt)]
                            # aggregate data
                            res = np.expand_dims(subeeg.transpose(), axis=0)
                            X = np.concatenate((X, res), axis=0)
                            y.extend(np.array([1]))
                            sampleID += 1
                            samplespike += 1
                            print(
                                "No. " + str(sampleID) + " (spike): " + filesub + " " + str(
                                    round(p / downrate, 2)) + "s.")
                        if contain_spikefile_in_bckg:
                            # available ranges
                            rangelist = chain(range(nt, math.floor(times[0] * downrate - nT * 2 + 1), nT),
                                              range(math.ceil(times[-1] * downrate + nT * 2), msm - nt + 1, nT))
                            if len(times) > 1:
                                for i in range(len(times) - 1):
                                    rangelist = chain(rangelist, range(math.ceil(times[i] * downrate + nT * 2),
                                                                       math.floor(times[i + 1] * downrate - nT * 2 + 1),
                                                                       nT))
                            bg_pos = np.random.choice(list(rangelist), size=round(len(times) * (1 - ratio) / ratio), replace=False)
                            for p in bg_pos:
                                p = round(p)
                                subeeg = eeg[:, (p - nt): (p + nt)]
                                # aggregate data
                                res = np.expand_dims(subeeg.transpose(), axis=0)
                                X = np.concatenate((X, res), axis=0)
                                y.extend(np.array([0]))
                                sampleID += 1
                                samplenormal += 1
                                print("No. " + str(sampleID) + " (background): " + filesub + " " + str(
                                    round(p / downrate, 2)) + "s.")
                    # if len(times) != 0:  # spike time, unit: second(s)
                    #     pos = np.round(times * downrate)
                    #     target = np.array([1])
                    # else:  # normal, randomly pick pos to crop
                    #     pos = np.random.choice(range(nt, msm, nT), size=randn, replace=False)
                    #     target = np.array([0])
                    # for p in pos:
                    #     p = round(p)
                    #     subeeg = eeg[:, (p - nt): (p + nt)]
                    #     # aggregate data
                    #     res = np.expand_dims(subeeg, axis=0)
                    #     X = np.concatenate((X, res), axis=0)
                    #     y.extend(target)
                    #     # ========================
                    #     sampleID += 1
                    #     if target[0] == 1:
                    #         samplespike += 1
                    #     else:
                    #         samplenormal += 1
                    #     if (samplespike / sampleID) < ratio:
                    #         break
                    # if (samplespike / sampleID) < ratio:
                    #     break

            randn = math.ceil((samplespike / ratio - sampleID) / len(normalfiledict))
            # randn = math.ceil((numspikefile * ((1 - ratio) / ratio)) / len(normalfilesub))

            # ==============================================
            if random:
                nfiledict = normalfiledict
            else:
                nfiledict = bckgfiledict

            if (not random) or (not contain_spikefile_in_bckg):

                for filesub in nfiledict.keys():
                    filename = osp.join(path, 'newEEGdata', filesub + '.m00')
                    if osp.exists(filename):
                        # read X
                        eeg = read_m00_file(filename)
                        eeg = preprocess_eeg(eeg)

                        # ---------------------------------------
                        # read y
                        times = np.array(nfiledict[filesub])
                        msm = eeg.shape[1]  # total number of measurements
                        if len(times) == 0:
                            continue
                        else:
                            pos = np.round(times * downrate)

                        # -----------------------------------------------
                        # if random:
                        #     pos = np.random.choice(range(nt, msm - nt + 1, nT), size=randn, replace=False)
                        for p in pos:
                            p = round(p)
                            subeeg = eeg[:, (p - nt): (p + nt)]
                            # aggregate data
                            res = np.expand_dims(subeeg.transpose(), axis=0)
                            X = np.concatenate((X, res), axis=0)
                            y.extend(np.array([0]))
                            # ========================
                            sampleID += 1
                            samplenormal += 1
                            print(
                                "No. " + str(sampleID) + " (normal): " + filesub + " " + str(round(p / downrate, 2)) + "s.")
                            if random and ((samplespike / sampleID) <= ratio):
                                break
                        if random and ((samplespike / sampleID) <= ratio):
                            break

            print('\n\n\n')
            print("The total number of samples is " + str(sampleID))
            print("The number of spike samples is " + str(samplespike))
            print("The number of non-spike samples is " + str(samplenormal))

            sys.stdout = original_stdout

            # =============WRITE END=================================

        y = np.array(y)

        # export
        output_path = dataDir + SSDname + "/seed" + str(seed)
        np.savez(output_path, X=X, y=y)

    else:

        output_path = dataDir + SSDname + "/seed" + str(seed) + '.npz'
        Y = np.load(output_path)
        y = Y['y']
        X = Y['X']

        yp = []

    print(X.shape)
    x = np.split(X, np.arange(batch_size, X.shape[0] + 1, batch_size))
    for i in range(len(x)):
        X_i = np.expand_dims(x[i], axis=2)
        print(X_i.shape)
        yp.extend(model.predict(X_i).flatten())

    yp = np.array(yp)

    savepath = dataDir + SSDname + '/output_seed' + str(seed) + '.npz'
    np.savez(savepath, outputm=yp, targetm=y)

    thres = 0.5

    yb = (yp > thres).astype(float)  # threshold
    recall = recall_score(y, yb, average='binary')  # recall = TP / (TP + FN), find completely
    prec = precision_score(y, yb, average='binary')  # precision = TP / (TP + FP), find accurately
    spec = specificity_score(y, yb, average='binary')
    f1 = f1_score(y, yb, average='binary')  # f1 score = 2 * precision * recall / (precision + recall)
    prauc = average_precision_score(y, yp)
    auc = roc_auc_score(y, yp)
    oos = {'recall': round(recall, 4),
           'prec': round(prec, 4),
           'spec': round(spec, 4),
           'f1': round(f1, 4),
           'prauc': round(prauc, 4),
           'auc': round(auc, 4)}
    REC.append(round(recall, 4))
    PREC.append(round(prec, 4))
    SPEC.append(round(spec, 4))
    F1.append(round(f1, 4))
    PRAUC.append(round(prauc, 4))
    AUC.append(round(auc, 4))
    index_col.append(seed)
    print(oos)

    if draw:
        print("Begin to plot:")
        savepdfpath = osp.join(dataDir + SSDname + "/seed" + str(seed) + '.pdf')
        pdf = backend_pdf.PdfPages(savepdfpath)
        for row in range(X.shape[0]):
            subeeg = X[row, :, :19].transpose()

            nd_ = subeeg.shape[0]
            nt_ = subeeg.shape[1]
            timent = np.arange(nt_)
            fig = plt.figure(figsize=(3, 6))
            gs = gridspec.GridSpec(nrows=nd_, ncols=2)

            for d in range(nd_):
                ax = fig.add_subplot(gs[d, 0])
                txt = ax.text(0.5, 0.5,
                              mono_channels[d] + '-avg',
                              size=8, ha='center', color='black')
                if d == 0:
                    ax.set_title('No.' + str(row), fontsize=8)
                ax.axis('off')
                ax = fig.add_subplot(gs[d, 1])
                ax.tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)
                # ax.plot(time, data[d, :], linewidth=1)  # maybe should be commented
                ax.plot(timent, subeeg[d, :], linewidth=0.8)
                if d == 0:
                    ax.set_title('y: ' + str(int(y[row])) + ', yp: ' + str(round(float(yp[row]), 2)), fontsize=8)
                ax.axis('off')

            pdf.savefig(fig)
            plt.clf()

        pdf.close()

ar = np.array([index_col, REC, PREC, SPEC, F1, PRAUC, AUC]).transpose()
print(ar)
np.savetxt(targetDir + "testTable_" + SSDname + '_' + str(thres) + ".csv", ar, fmt='%.4f', delimiter=",",
           header="seed,recall,prec,spec,f1,prauc,auc")

# # export
# output_path = targetDir + "SSD_btheeg"
# np.savez(output_path, y=y, yp=yp)
