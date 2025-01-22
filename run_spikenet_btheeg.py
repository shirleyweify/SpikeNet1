import os

import mne.io
import numpy as np
import scipy.io as sio
import hdf5storage as hs
from tqdm import tqdm
from mne.filter import notch_filter, filter_data
from keras.models import model_from_json

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
batch_size = 1000


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
    sourceDir = "/Users/shirleywei/Dropbox/Data/Spike/newEEGdata/"
    targetDir = "/Users/shirleywei/Dropbox/Data/Spike/newEEGdata_jama/"
    if not os.path.exists(targetDir):
        os.makedirs(targetDir)
    files = os.listdir(sourceDir)

    # load model 
    with open("model/spikenet1.o_structure.txt", "r") as ff:
        json_string = ff.read()

    model = model_from_json(json_string)
    model.load_weights("model/spikenet1.o_weights.h5")

    for fn in files[:1]:
        print("--scan " + fn)

        # read data
        input_path = sourceDir + fn
        data = read_raw_file(input_path)

        # preprocess            
        eeg = preprocess_eeg(data)

        # run model
        start_ids = np.arange(0, eeg.shape[1] - L + 1, step)
        start_ids = np.array_split(start_ids, int(np.ceil(len(start_ids) * 1. / batch_size)))
        yp = []
        for startid in tqdm(start_ids, leave=False):
            X = eeg[:, list(map(lambda x: np.arange(x, x + L), startid))].transpose(1, 2, 0)
            X = np.expand_dims(X, axis=2)
            yp.extend(model.predict(X).flatten())

        yp = np.array(yp)
        padleft = (eeg.shape[1] - len(yp) * step) // 2
        padright = eeg.shape[1] - len(yp) * step - padleft
        yp = np.r_[np.zeros(padleft) + yp[0], np.repeat(yp, step, axis=0), np.zeros(padright) + yp[-1]]
        print(yp)

        # export
        output_path = targetDir + "SSD_" + fn
        # np.save(output_path, yp)
        # sio.savemat(output_path, {'yp': yp})
