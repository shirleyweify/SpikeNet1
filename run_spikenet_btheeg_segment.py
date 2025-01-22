import os

import mne.io
import numpy as np
import scipy.io as sio
import hdf5storage as hs
from tqdm import tqdm
from mne.filter import notch_filter, filter_data
from keras.models import model_from_json
import pandas as pd

from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score
from imblearn.metrics import specificity_score

# global var
notch_freq = 60
bp_freq = [0.5, None]
# mono_channels = ['FP1', 'F3', 'C3', 'P3', 'F7', 'T3', 'T5', 'O1', 'FZ', 'CZ', 'PZ', 'FP2', 'F4', 'C4', 'P4', 'F8', 'T4',
#                  'T6', 'O2']
# bipolar_channels = ['FP1-F7', 'F7-T3', 'T3-T5', 'T5-O1', 'FP2-F8', 'F8-T4', 'T4-T6', 'T6-O2', 'FP1-F3', 'F3-C3',
#                     'C3-P3', 'P3-O1', 'FP2-F4', 'F4-C4', 'C4-P4', 'P4-O2', 'FZ-CZ', 'CZ-PZ']
mono_channels = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'Fz',
                 'Cz', 'Pz']
bipolar_channels = ['Fp1-F7', 'F7-T3', 'T3-T5', 'T5-O1', 'Fp2-F8', 'F8-T4', 'T4-T6', 'T6-O2', 'Fp1-F3', 'F3-C3',
                    'C3-P3', 'P3-O1', 'Fp2-F4', 'F4-C4', 'C4-P4', 'P4-O2', 'Fz-Cz', 'Cz-Pz']
Fs = 128
L = int(round(1 * Fs))
step = 1
batch_size = 64


def read_mat_file(path):
    # read data
    try:
        res = hs.loadmat(path)
    except Exception as ee:
        res = sio.loadmat(path)

    seg = res['data'][0:19, :]
    res['data'] = np.where(np.isnan(seg), 0, seg)

    # montages
    bipolar_ids = np.array(
        [[mono_channels.index(bc.split('-')[0]), mono_channels.index(bc.split('-')[1])] for bc in bipolar_channels])
    bipolar_data = res['data'][bipolar_ids[:, 0]] - res['data'][bipolar_ids[:, 1]]
    average_data = res['data'] - res['data'].mean(axis=0);
    res['data'] = np.concatenate([average_data, bipolar_data], axis=0)

    return res

def read_raw_file(path):
    # read data
    f = open(path, "r")
    txt = f.readlines()[0]
    res = np.loadtxt(path, skiprows=2)
    res = res.transpose()

    seg = res[0:19, :]
    res = np.where(np.isnan(seg), 0, seg)

    # montages
    bipolar_ids = np.array(
        [[mono_channels.index(bc.split('-')[0]), mono_channels.index(bc.split('-')[1])] for bc in bipolar_channels])
    bipolar_data = res[bipolar_ids[:, 0]] - res[bipolar_ids[:, 1]]
    average_data = res - res.mean(axis=0)
    res = np.concatenate([average_data, bipolar_data], axis=0)

    return res


def preprocess_eeg(X):
    # Notch and highpass
    X = filter_data(X, Fs, bp_freq[0], bp_freq[1], n_jobs=-1, method='fir', verbose=False)
    X = notch_filter(X, Fs, notch_freq, n_jobs=-1, method='fir', verbose=False)

    return X


if __name__ == '__main__':

    # I/O directories
    sourceDir = "/Users/shirleywei/Dropbox/Data/Spike/newEEGdata_resample256Hz/"
    targetDir = "Output/"
    if not os.path.exists(targetDir):
        os.makedirs(targetDir)
    files = os.listdir(sourceDir)

    # load model 
    with open("model/spikenet1.o_structure.txt", "r") as ff:
        json_string = ff.read()

    model = model_from_json(json_string)
    model.load_weights("model/spikenet1.o_weights.h5")

    y, yp = [], []
    X = np.array([]).reshape(0, 128, 37)

    for fn in files:
        print("--scan " + fn)

        # read data
        input_path = sourceDir + fn
        data = np.load(input_path)
        eeg = data['eeg']
        y.extend(data['target'])

        # montages
        bipolar_ids = np.array(
            [[mono_channels.index(bc.split('-')[0]), mono_channels.index(bc.split('-')[1])] for bc in bipolar_channels])
        bipolar_data = eeg[bipolar_ids[:, 0]] - eeg[bipolar_ids[:, 1]]
        average_data = eeg - eeg.mean(axis=0)
        res = np.concatenate([average_data, bipolar_data], axis=0)
        res = np.expand_dims(res.transpose(), axis=0)

        # aggregate data
        X = np.concatenate((X, res), axis=0)

    print(X.shape)
    x = np.split(X, np.arange(batch_size, len(files)+1, batch_size))
    for i in range(len(x)):
        X = np.expand_dims(x[i], axis=2)
        yp.extend(model.predict(X).flatten())

    y = np.array(y)
    yp = np.array(yp)
    print(y), print(yp)

    yb = (yp > 0.1).astype(float)  # threshold
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
    oos_ = pd.DataFrame(oos, index=[0])
    print(oos_)

    # export
    output_path = targetDir + "SSD_btheeg_segment"
    np.save(output_path, yp)
