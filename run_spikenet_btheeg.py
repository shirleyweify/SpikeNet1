import numpy as np
import os
import os.path as osp
import pandas as pd
import math
from mne.filter import notch_filter, filter_data, resample
from keras.models import model_from_json
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score
from imblearn.metrics import specificity_score

seed_range = [1, 2]

# # settings
# np.random.seed(1)

# path = osp.join('C:', osp.sep, 'Users', 'FBE', 'Dropbox', 'Data', 'Spike')
path = osp.join(osp.sep, 'Users', 'shirleywei', 'Dropbox', 'Data', 'Spike')
files = os.listdir(osp.join(path, 'newEEGdata'))
filesubs = [f[:-4] for f in files]

test_model = 'tuev'  # 'meg' or 'tuev'
calcu_ratio = False
chavg = False

# global var
notch_freq = 60
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
batch_size = 1000


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
ratio = 0.37455223679603544  # 1/0 ratio according to MEG data

# -----------------
nT = nX + nZ * 2
nt = int(nT / 2)  # half length
sampleID = 0
samplespike = 0
samplenormal = 0
nrows = []

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

# create spike file dict
filedict = dict()  # 'filesub': [time1, time2, ...] (seconds) if time is empty then normal subs
spikefile = spikefile[spikefile['comments'].isna()]  # remove red rows (not sure)
spikefilesub = spikefile['interictal_file'].dropna().unique().tolist()
numspikefile = 0
for f in spikefilesub:
    pd = spikefile[spikefile['interictal_file'] == f]
    extime = pd['exact_time'].unique().tolist()
    sec = [t.minute * 60 + t.second + t.microsecond / 1e6 for t in extime]  # convert to seconds
    filedict[f] = sec
    numspikefile += len(sec)
for f in normalfilesub:
    filedict[f] = list()

# spike files and normal ones
randn = math.ceil((numspikefile * ((1 - ratio) / ratio)) / len(normalfilesub))

# ==============================================

# I/O directories
targetDir = "Output/"
if not os.path.exists(targetDir):
    os.makedirs(targetDir)

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

for seed in range(seed_range[0], seed_range[1] + 1):
    np.random.seed(seed)

    # initialization
    y, yp = [], []
    X = np.array([]).reshape(0, 128, 37)

    # ==============================================
    for filesub in filedict.keys():
        filename = osp.join(path, 'newEEGdata', filesub + '.m00')
        if osp.exists(filename):
            # read X
            data = read_m00_file(filename)
            eeg = preprocess_eeg(data)

            # ---------------------------------------
            # read y
            times = np.array(filedict[filesub])
            msm = eeg.shape[1]  # total number of measurements
            if len(times) != 0:  # spike time, unit: second(s)
                pos = np.round(times * downrate)
                target = np.array([1])
            else:  # normal, randomly pick pos to crop
                pos = np.random.choice(range(nt, msm, nT), size=randn, replace=False)
                target = np.array([0])
            for p in pos:
                p = round(p)
                subeeg = eeg[:, (p - nt): (p + nt)]
                # aggregate data
                res = np.expand_dims(subeeg.transpose(), axis=0)
                X = np.concatenate((X, res), axis=0)
                y.extend(target)
                # ========================
                sampleID += 1
                if target[0] == 1:
                    samplespike += 1
                else:
                    samplenormal += 1
                if (samplespike / sampleID) < ratio:
                    break
            if (samplespike / sampleID) < ratio:
                break

    print(X.shape)
    x = np.split(X, np.arange(batch_size, sampleID + 1, batch_size))
    for i in range(len(x)):
        X = np.expand_dims(x[i], axis=2)
        yp.extend(model.predict(X).flatten())

    y = np.array(y)
    yp = np.array(yp)
    # print(y), print(yp)

    yb = (yp > 0.25).astype(float)  # threshold
    recall = recall_score(y, yb, average='binary')  # recall = TP / (TP + FN), find completely
    prec = precision_score(y, yb, average='binary')  # precision = TP / (TP + FP), find accurately
    spec = specificity_score(y, yb, average='binary')
    f1 = f1_score(y, yb, average='binary')  # f1 score = 2 * precision * recall / (precision + recall)
    prauc = average_precision_score(y, yp)
    auc = roc_auc_score(y, yp)
    # oos = {'recall': round(recall, 4),
    #        'prec': round(prec, 4),
    #        'spec': round(spec, 4),
    #        'f1': round(f1, 4),
    #        'prauc': round(prauc, 4),
    #        'auc': round(auc, 4)}
    REC.append(round(recall, 4))
    PREC.append(round(prec, 4))
    SPEC.append(round(spec, 4))
    F1.append(round(f1, 4))
    PRAUC.append(round(prauc, 4))
    AUC.append(round(auc, 4))
    # print(oos)

    # export
    output_path = targetDir + "SSD_btheeg/SSD_btheeg_seed" + str(seed)
    np.savez(output_path, y=y, yp=yp)

# print(oos)
# res = pd.DataFrame(oos)
# res = pd.DataFrame(data={
#     'recall': REC,
#     'prec': PREC,
#     'spec': SPEC,
#     'f1': F1,
#     'prauc': PRAUC,
#     'auc': AUC
# })
# print(res)
# res.to_csv(targetDir + "testTable_SSD_btheeg.csv")

ar = np.array([REC, PREC, SPEC, F1, PRAUC, AUC])
print(ar)
np.savetxt(targetDir + "testTable_SSD_btheeg.csv", ar,
           delimiter=",", header="Recall, Prec, Spec, F1, PRAUC, AUC")

# # export
# output_path = targetDir + "SSD_btheeg"
# np.savez(output_path, y=y, yp=yp)
