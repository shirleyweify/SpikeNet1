import mne
import os
import os.path as osp
import numpy as np
import pandas as pd
import sys
import time
from mne.filter import notch_filter, filter_data, resample
from keras.models import model_from_json
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score
from imblearn.metrics import specificity_score

from spikePath import *

start = time.time()

dryrun = False  # if True, test on small samples

np.random.seed(1)

# =========================================================
# I/O directories
targetDir = "Output/"
if not os.path.exists(targetDir):
    os.makedirs(targetDir)

# =======================================================
# sys settings

os_sys = 'osx'  # 'osx' or 'win'
data_type = 'eval'  # 'train' or 'eval'
read_type = 'wd'  # 'wd', 'seagate' or 'dropbox'
write_type = 'wd'  # 'wd', 'seagate' or 'dropbox'
spike_type = 'IED'

# Label type

labeling = 'tuev41eeg500hf45spikenet'

# =========================================================
# path

dir_dict = data_path(os_sys=os_sys, read_type=read_type, write_type=write_type,
                     spike_type=spike_type)
work_dir_dict = work_path(os_sys=os_sys, spike_type=spike_type, label_type=labeling)
# Combine read path
read_data_path = osp.join(dir_dict['read_dir'], data_type)
# Combine write path
train_save_root = osp.join(dir_dict['write_dir'], labeling)
# if detrend:
#     train_save_root = train_save_root + '_detrend'  # if detrend, add in filenames
train_save_path = osp.join(train_save_root, data_type)  # 'train' or 'eval'
os.makedirs(train_save_path, exist_ok=True)

# -------------------------------
# global var
notch_freq = 60
bp_freq = [0.5, None]
org_channels = ['EEG FP1-REF', 'EEG F3-REF', 'EEG C3-REF', 'EEG P3-REF',
                'EEG F7-REF', 'EEG T3-REF', 'EEG T5-REF', 'EEG O1-REF',
                'EEG FZ-REF', 'EEG CZ-REF', 'EEG PZ-REF', 'EEG FP2-REF',
                'EEG F4-REF', 'EEG C4-REF', 'EEG P4-REF', 'EEG F8-REF',
                'EEG T4-REF', 'EEG T6-REF', 'EEG O2-REF']
mono_channels = ['FP1', 'F3', 'C3', 'P3', 'F7', 'T3', 'T5', 'O1', 'FZ', 'CZ', 'PZ', 'FP2', 'F4', 'C4', 'P4', 'F8', 'T4',
                 'T6', 'O2']
bipolar_channels = ['FP1-F7', 'F7-T3', 'T3-T5', 'T5-O1', 'FP2-F8', 'F8-T4', 'T4-T6', 'T6-O2', 'FP1-F3', 'F3-C3',
                    'C3-P3', 'P3-O1', 'FP2-F4', 'F4-C4', 'C4-P4', 'P4-O2', 'FZ-CZ', 'CZ-PZ']
Fs = 128
L = int(round(1 * Fs))
step = 1
batch_size = 1000

# =========================================================
# parameters
downrate = Fs
nX = 128  # = T
nZ = 0  # = p / 2
sampleID = 0
subID = 0

# ==============================================

times = 10

time_start_end = np.array([]).reshape(0, 2)
filename = []

# ======== TUH settings

channel_anode = [
    'EEG FP1-REF', 'EEG F7-REF', 'EEG T3-REF', 'EEG T5-REF',
    'EEG FP2-REF', 'EEG F8-REF', 'EEG T4-REF', 'EEG T6-REF',
    'EEG A1-REF', 'EEG T3-REF', 'EEG C3-REF', 'EEG CZ-REF', 'EEG C4-REF', 'EEG T4-REF',
    'EEG FP1-REF', 'EEG F3-REF', 'EEG C3-REF', 'EEG P3-REF',
    'EEG FP2-REF', 'EEG F4-REF', 'EEG C4-REF', 'EEG P4-REF'
]
channel_cathode = [
    'EEG F7-REF', 'EEG T3-REF', 'EEG T5-REF', 'EEG O1-REF',
    'EEG F8-REF', 'EEG T4-REF', 'EEG T6-REF', 'EEG O2-REF',
    'EEG T3-REF', 'EEG C3-REF', 'EEG CZ-REF', 'EEG C4-REF', 'EEG T4-REF', 'EEG A2-REF',
    'EEG F3-REF', 'EEG C3-REF', 'EEG P3-REF', 'EEG O1-REF',
    'EEG F4-REF', 'EEG C4-REF', 'EEG P4-REF', 'EEG O2-REF'
]
new_channel_names = [
    'FP1-F7', 'F7-T3', 'T3-T5', 'T5-O1',
    'FP2-F8', 'F8-T4', 'T4-T6', 'T6-O2',
    'A1-T3', 'T3-C3', 'C3-CZ', 'CZ-C4', 'C4-T4', 'T4-A2',
    'FP1-F3', 'F3-C3', 'C3-P3', 'P3-O1',
    'FP2-F4', 'F4-C4', 'C4-P4', 'P4-O2'
]
channel_names = list(set(channel_anode + channel_cathode))  # all channels we require for the raw data

# =================

# parameters
downrate_ = 250  # downsample to 250Hz (TUEV sfreq is 250 Hz)
nX_ = 250  # = T
nZ_ = 125  # = p / 2
freq = [1, 45]  # default: [1, 45]


def read_edf_file(path):
    # read data
    raw = mne.io.read_raw_edf(path)
    data = raw.copy().get_data()
    chnames = raw.ch_names
    sfreq = raw.info['sfreq']
    downrate = Fs

    seg = data[:, :] * 1e7
    data = np.where(np.isnan(seg), 0, seg)
    data = - data

    # switch rows
    switch_idx = [chnames.index(c) for c in org_channels]
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


# =====================================================
# load path
# read_data_path = osp.join('/Users/shirleywei/Dropbox/Data/Spike/tuev_v2.0.0/edf', data_type)

# subject_id = os.listdir(read_data_path)
subject_id = [id for id in os.listdir((read_data_path)) if not id.startswith('.')]
# if '.DS_Store' in subject_id:
#     subject_id.remove('.DS_Store')
if dryrun:
    subject_id = subject_id[:3]

# ----------

for subject in subject_id:  # subject = '00000021'
    try:

        subject_path = osp.join(read_data_path, subject)
        subject_experiment_id = [ex[:-4] for ex in os.listdir(subject_path) if 'edf' in ex]  # ['00000021_00000001']
        spike_type_per_patient = np.array([])

        y, yp = [], []
        X = np.array([]).reshape(0, 128, 37)
        X_ = np.array([]).reshape(0, 500, 41)
        time_start_end = np.array([]).reshape(0, 2)
        filename = []

        print('--------------------')
        print('Starting to read...')

        for sub_experiment in subject_experiment_id:  # sub_experiment = '00000021_00000001'
            # experiment = sub_experiment[9:]  # '00000001'
            edf_filepath = osp.join(subject_path, sub_experiment + '.edf')
            rec_filepath = osp.join(subject_path, sub_experiment + '.rec')

            # read raw edf EEG data
            data = read_edf_file(edf_filepath)
            # read raw edf EEG data ===================
            rawdata = mne.io.read_raw_edf(edf_filepath)

            eeg = preprocess_eeg(data)

            # ==========================================

            data = rawdata.copy().get_data()
            raw_channels = rawdata.ch_names
            # create montage
            order_anode = [raw_channels.index(ch) for ch in channel_anode]
            order_cathode = [raw_channels.index(ch) for ch in channel_cathode]
            order_org = [raw_channels.index(ch) for ch in org_channels]
            data_average = data[order_org] - data[order_org].mean(axis=0)
            data_tcp22montage = data[order_anode] - data[order_cathode]
            montage = np.concatenate((data_average, data_tcp22montage), axis=0)  # 19 + 22 = 41 channels

            # preprocess
            sfreq = rawdata.info['sfreq']
            montage = -montage
            montage = mne.filter.resample(
                montage, up=1, down=sfreq / downrate_
            )  # resample
            montage = mne.filter.filter_data(
                montage,
                sfreq=sfreq, l_freq=freq[0], h_freq=freq[1]
            ) * 1e7  # filter
            montage = mne.filter.notch_filter(montage, Fs=downrate_, freqs=60)
            subeeg = mne.epochs.detrend(montage)  # detrend

            # ===========================================
            nt = subeeg.shape[1]
            ns = int(np.floor((nt - downrate_) / downrate_))
            for s in range(ns):
                t_start = s + 0.5  # seconds
                t_end = s + 1.5  # seconds
                # ---------- spikenet
                nt_start = round(t_start * Fs)  # round to integer
                nt_end = round(t_end * Fs)
                t0_start = nt_start - nZ
                t0_end = nt_end + nZ
                # -------- ndl
                nt_start = round(t_start * downrate_)  # round to integer
                nt_end = round(t_end * downrate_)
                t0_start_ = nt_start - nZ_
                t0_end_ = nt_end + nZ_
                # ------------
                if t0_start_ >= 0 and t0_end_ <= eeg.shape[1]:
                    eegseg = eeg[:, t0_start: t0_end]
                    eegseg_ = subeeg[:, t0_start_: t0_end_]
                    res = np.expand_dims(eegseg.transpose(), axis=0)
                    res_ = np.expand_dims(eegseg_.transpose(), axis=0)
                    X = np.concatenate((X, res), axis=0)
                    X_ = np.concatenate((X_, res_), axis=0)
                    # y.extend(target)

                    sampleID += 1

                    time_start_end = np.concatenate((time_start_end, np.array([t_start, t_end]).reshape(1, 2)),
                                                    axis=0)
                    filename.append(sub_experiment)

        print('--------------------')
        print('Predicting...')

        # load model
        with open("model/spikenet1.o_structure.txt", "r") as ff:
            json_string = ff.read()

        model = model_from_json(json_string)
        model.load_weights("model/spikenet1.o_weights.h5")

        print(X.shape), print(X_.shape)
        n_sample_i = X_.shape[0]

        x = np.split(X, np.arange(batch_size, n_sample_i + 1, batch_size))
        for i in range(len(x)):
            X = np.expand_dims(x[i], axis=2)
            yp.extend(model.predict(X).flatten())

        print(len(yp)), print(len(time_start_end)), print(len(filename))

        # y = np.array(y)
        yp = np.array(yp)
        filename = np.array(filename)

        for k in range(n_sample_i):
            eegdata = X_[k, :, :].transpose()
            sample_name = str(int(sampleID - n_sample_i + k)) + '.npz'
            print(sample_name)
            np.savez(osp.join(train_save_path, sample_name),
                     eeg=eegdata, target=np.array([yp[k]]),
                     time=time_start_end[k, :], filename=np.array([filename[k]]))

        subID += 1
        print('--------------------')
        print('Subject id ' + str(subID) + ' done')
        print('====================')

    except Exception as error:
        print("An error occurred:", error)
        continue  # force to start the next iteration

print("The total number of subjects is " + str(subID))
print("The total number of samples is " + str(sampleID))

end = time.time()

print('The execution time of the program is ' + str(round((end - start) / 60)) + ' minute(s).')
