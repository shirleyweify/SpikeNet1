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

start = time.time()

dryrun = False  # if True, test on small samples

# =======================================================
# sys settings
# data_type = 'eval'  # 'train' or 'eval'
data_type = ['train', 'eval']  # for loop

np.random.seed(1)

# =========================================================
# I/O directories
targetDir = "Output/"
if not os.path.exists(targetDir):
    os.makedirs(targetDir)

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

# ==============================================

# initialization
y, yp = [], []
X = np.array([]).reshape(0, 128, 37)

time_start_end = np.array([]).reshape(0, 2)
filename_ls = []

def read_edf_file(path):
    # read data
    raw = mne.io.read_raw_edf(path, verbose=False)
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

arr = np.load("/Users/shirleywei/Dropbox/Projects/NestedDeepLearningModel/GitCodesMac/AIcode/Output/TUH_TEST_LOCAL.npy")
datapath = "/Users/shirleywei/Dropbox/Data/Spike/tuh22eeg500hf70/"

for fileindex in arr:
    ar = np.load(datapath + str(fileindex) + '.npz')
    filename = ar['filename'][0]
    time = ar['time']
    target = ar['target']
    # print(filename), print(time), print(target)
    if fileindex < 10000:
        # ar = np.load(datapath + 'train/' + str(fileindex) + '.npz')
        # filename = ar['filename'][0]
        # time = ar['time']
        # target = ar['target']
        read_data_path = '/Users/shirleywei/Dropbox/Data/Spike/tuev_v2.0.0/edf/train'
        subject_path = osp.join(read_data_path, filename[:8])
        edf_filepath = osp.join(subject_path, filename + '.edf')
    else:
        # ar = np.load(datapath + 'eval/' + str(fileindex) + '.npz')
        # filename = ar['filename'][0]
        # time = ar['time']
        # target = ar['target']
        read_data_path = '/Users/shirleywei/Dropbox/Data/Spike/tuev_v2.0.0/edf/eval'
        subject_path = osp.join(read_data_path, filename[5:8])
        edf_filepath = osp.join(subject_path, filename + '.edf')

    # read data
    try:
        data = read_edf_file(edf_filepath)
    except Exception as error:
        print("An error occurred:", error)
        continue  # force to start the next iteration
    # process data
    eeg = preprocess_eeg(data)
    # load time
    t_start = time[0]
    t_end = time[1]
    nt_start = round(t_start * Fs)  # round to integer
    nt_end = round(t_end * Fs)
    t0_start = nt_start - nZ
    t0_end = nt_end + nZ
    ts_start = nt_start - int(Fs / 2)  # leave .5 seconds left
    ts_end = nt_end + int(Fs / 2)  # leave .5 seconds right
    if ts_start >= 0 and ts_end <= eeg.shape[1]:
        subeeg = eeg[:, t0_start: t0_end]
        # aggregate data
        res = np.expand_dims(subeeg.transpose(), axis=0)
        X = np.concatenate((X, res), axis=0)
        y.extend(target)

        time_start_end = np.concatenate((time_start_end, ar['time'].reshape(1, 2)), axis=0)
        filename_ls.append(filename)

        sampleID += 1
        print("No. " + str(sampleID) + " done.")



# for dt in data_type:
#
#     # load path
#     read_data_path = osp.join('/Users/shirleywei/Dropbox/Data/Spike/tuev_v2.0.0/edf', dt)
#
#     # subject_id = os.listdir(read_data_path)
#     subject_id = [id for id in os.listdir((read_data_path)) if not id.startswith('.')]
#     # if '.DS_Store' in subject_id:
#     #     subject_id.remove('.DS_Store')
#     if dryrun:
#         subject_id = subject_id[:10]
#
#
#     # ----------
#
#     for subject in subject_id:  # subject = '00000021'
#         subject_path = osp.join(read_data_path, subject)
#         subject_experiment_id = [ex[:-4] for ex in os.listdir(subject_path) if 'edf' in ex]  # ['00000021_00000001']
#         spike_type_per_patient = np.array([])
#         for sub_experiment in subject_experiment_id:  # sub_experiment = '00000021_00000001'
#             # experiment = sub_experiment[9:]  # '00000001'
#             edf_filepath = osp.join(subject_path, sub_experiment + '.edf')
#             rec_filepath = osp.join(subject_path, sub_experiment + '.rec')
#             try:
#                 # read raw edf EEG data
#                 data = read_edf_file(edf_filepath)
#             except Exception as error:
#                 print("An error occurred:", error)
#                 continue  # force to start the next iteration
#
#             eeg = preprocess_eeg(data)
#
#             # read rec file
#             rec_txt = np.loadtxt(rec_filepath, delimiter=',')
#             cond_label = rec_txt[:, 3] <= 3  # 6 spikes to 1/0 variable
#             cond_label = cond_label.astype('int')  # change bool values to integers (1/0)
#             rec_txt = np.c_[rec_txt, cond_label]  # [ch_loc, t_start, t_end, 1-6_types, 0/1_lab]
#             rec_txt = np.round(rec_txt, 1)  # np.round: round to float
#             rec_txt = rec_txt[rec_txt[:, 1].argsort()]  # ascending sort by the second column (t_start)
#
#             # remove 90% overlapping time periods
#             # not necessary after round time to 0.1, largest overlapping = 0.9 / 1.1 < 90%
#             rec_time = rec_txt[:, 1:3]  # select two columns of time
#             rec_time = np.unique(rec_time, axis=0)  # select unique periods
#
#             # segment data
#             for rec in range(rec_time.shape[0]):
#                 t_start = rec_time[rec, 0]
#                 t_end = rec_time[rec, 1]
#                 nt_start = round(t_start * Fs)  # round to integer
#                 nt_end = round(t_end * Fs)
#                 t0_start = nt_start - nZ
#                 t0_end = nt_end + nZ
#                 ts_start = nt_start - int(Fs/2)  # leave .5 seconds left
#                 ts_end = nt_end + int(Fs/2)  # leave .5 seconds right
#                 if ts_start >= 0 and ts_end <= eeg.shape[1]:
#                     subeeg = eeg[:, t0_start: t0_end]
#                     cond_channel = np.where((rec_txt[:, 1] == t_start)
#                                             & (rec_txt[:, 2] == t_end))
#                     info = rec_txt[cond_channel, :][0]  # select info during the period
#                     spike_type = np.unique(info[:, -2])  # 1-6 unique types in the 1s epoch
#                     spike_type_per_patient = np.concatenate((spike_type_per_patient, spike_type))
#                     # target var
#                     target1 = max(info[:, -1])  # 1 for containing IEDs
#                     target0 = min(info[:, -1])  # 0 for no IEDs
#                     target = np.array([target1])
#                     # aggregate data
#                     res = np.expand_dims(subeeg.transpose(), axis=0)
#                     X = np.concatenate((X, res), axis=0)
#                     y.extend(target)
#
#                     sampleID += 1
#
#                     time_start_end_ = np.array([t_start, t_end]).reshape(1, 2)
#                     time_start_end = np.concatenate((time_start_end, time_start_end_), axis=0)
#                     filename.append(sub_experiment)


# load model
with open("model/spikenet1.o_structure.txt", "r") as ff:
    json_string = ff.read()

model = model_from_json(json_string)
model.load_weights("model/spikenet1.o_weights.h5")

print(X.shape)
x = np.split(X, np.arange(batch_size, sampleID + 1, batch_size))
for i in range(len(x)):
    X = np.expand_dims(x[i], axis=2)
    yp.extend(model.predict(X).flatten())

y = np.array(y)
yp = np.array(yp)
filename = np.array(filename)
print(y), print(yp)
print(X.shape), print(len(y)), print(len(yp))

yb = (yp > 0.5).astype(float)  # threshold
recall = recall_score(y, yb, average='binary')  # recall = TP / (TP + FN), find completely
prec = precision_score(y, yb, average='binary')  # precision = TP / (TP + FP), find accurately
spec = specificity_score(y, yb, average='binary')
f1 = f1_score(y, yb, average='binary')  # f1 score = 2 * precision * recall / (precision + recall)
prauc = average_precision_score(y, yp)
auc = roc_auc_score(y, yp)
oos = {'recall': round(recall, 6),
       'prec': round(prec, 6),
       'spec': round(spec, 6),
       'f1': round(f1, 6),
       'prauc': round(prauc, 6),
       'auc': round(auc, 6)}
print(oos)

savename = "SpikeNet_eval_TUHhf45_seed8_npseed2029"

res = pd.DataFrame(oos)
res.to_csv("Output/" + savename + ".csv", index=False)

output_path = targetDir + savename + ".npz"

# # export
# if len(data_type) == 1:
#     output_path = targetDir + "SSD_tuh_" + data_type[0]
# elif len(data_type) == 2:
#     output_path = targetDir + "SSD_tuh_" + data_type[0] + "_" + data_type[1]
# else:
#     raise Exception("Error.")

np.savez(output_path, y=y, yp=yp, time=time_start_end, filename=filename_ls)

end = time.time()

print('The execution time of the program is ' + str(round((end - start) / 60)) + ' minute(s).')
