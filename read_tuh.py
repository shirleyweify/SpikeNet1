import mne
import os
import os.path as osp
import numpy as np
import pandas as pd
import sys
import time

from spikePath import *

start = time.time()

dryrun = False  # if True, test on small samples and do not save
save = False  # if True, save npz files

# =======================================================
# sys settings

os_sys = 'win'  # 'osx' or 'win'
data_type = 'train'  # 'train' or 'eval'
read_type = 'seagate'  # 'wd', 'seagate' or 'dropbox'
write_type = 'dropbox'  # 'wd', 'seagate' or 'dropbox'
spike_type = 'IED'

# Label type

labeling = 'tuev22eeg500'
# detrend = False  # whether to detrend or not (by linear trend)
# UPD V
# 'tuev22eeg500': dataset + number of channels + data type + dimension of measurements (22 x 500)
# OLD V
# IED_others: Y = 1 if contains IEDs (any one of 1/2/3); Y = 0 if other events (not contain any one of 1/2/3)
# containIED_others: Y = 1 if contains IEDs (1/2/3 with or w/o 4/5/6); Y = 0 if others (4/5/6)
# IED_nonIED: Y = 1 if only IED (1/2/3); Y = 0 if only non-IED (4/5/6)
# This is to differentiate IED from non-IED events
# IEDbckg_bckg: Y = 1 if IED with/without bckg (1/2/3 w/o 6); Y = 0 if only bckg (6)
# This is to identify IED (1/2/3) from background noises

# =========================================================
# path

dir_dict = data_path(os_sys=os_sys, read_type=read_type, write_type=write_type,
                     spike_type=spike_type)
# Combine read path
read_data_path = osp.join(dir_dict['read_dir'], data_type)
# Combine write path
train_save_root = osp.join(dir_dict['write_dir'], labeling)
# if detrend:
#     train_save_root = train_save_root + '_detrend'  # if detrend, add in filenames
train_save_path = osp.join(train_save_root, data_type)  # 'train' or 'eval'
os.makedirs(train_save_path, exist_ok=True)

# Customize montage

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

# subject_id = os.listdir(read_data_path)
subject_id = [id for id in os.listdir((read_data_path)) if not id.startswith('.')]
# if '.DS_Store' in subject_id:
#     subject_id.remove('.DS_Store')
if dryrun:
    subject_id = subject_id[:3]

n_channels = []  # number of channels of the montage (should be 22)
error_edf = []  # paths of unreadable edf files
overlap_exp = []  # with 90% overlapping time periods
time_rm_exp = []  # exp removed with time points not equals 500
time_prob_exp = []  # exp remianed with time points not equals 500
seg_num_lab = []  # segments with both 1 and 0 labels
target_list = []  # to compute the 0/1 ratio
downsample_list = []  # samples which are downrated

n_subjects = 0
n_sub_experiments = 0
n_experiments_avail = 0

count_overlap = 0
count_IED_nonIED = 0  # nsamples contain both IED & non-IED events
count_IED_bckg = 0  # nsamples contain both IED & bckg
count_eyem_artf = 0  # nsamples contain eyem or artf

# # number of samples that contain spsw/gped/pled/eyem/artf/bckg
# n_spsw = 0
# n_gped = 0
# n_pled = 0
# n_eyem = 0
# n_artf = 0
# n_bckg = 0

# number of patients that contain spsw/gped/pled/eyem/artf/bckg
n_spsw = 0
n_gped = 0
n_pled = 0
n_eyem = 0
n_artf = 0
n_bckg = 0

# number of channels that contain spsw/gped/pled/eyem/artf/bckg annotatioins
n_ch_spsw = []
n_ch_gped = []
n_ch_pled = []
n_ch_eyem = []
n_ch_artf = []
n_ch_bckg = []

# ch22_spsw = []  # samples with type 1 label on 22 channels
# ch22_gped = []  # samples with type 2 label on 22 channels
# ch22_pled = []  # samples with type 3 label on 22 channels
# ch22_eyem = []  # samples with type 4 label on 22 channels
# ch22_artf = []  # samples with type 5 label on 22 channels
# ch22_bckg = []  # samples with type 6 label on 22 channels

if data_type == 'train':
    sample_id = 0
elif data_type == 'eval':
    sample_id = 10000
else:
    raise Exception("Data type mis-specified.")

# parameters
downrate = 250  # downsample to 250Hz (TUEV sfreq is 250 Hz)
nX = 250  # = T
nZ = 125  # = p / 2

# [seg_ID, channel, 1-6 types]
table = np.array([['seg_id', 'ch_loc', 'lab_type']])

# ----------

for subject in subject_id:  # subject = '00000021'
    subject_path = osp.join(read_data_path, subject)
    subject_experiment_id = [ex[:-4] for ex in os.listdir(subject_path) if 'edf' in ex]  # ['00000021_00000001']
    n_subjects += 1
    spike_type_per_patient = np.array([])
    for sub_experiment in subject_experiment_id:  # sub_experiment = '00000021_00000001'
        # experiment = sub_experiment[9:]  # '00000001'
        edf_filepath = osp.join(subject_path, sub_experiment + '.edf')
        rec_filepath = osp.join(subject_path, sub_experiment + '.rec')
        n_sub_experiments += 1
        try:
            # read raw edf EEG data
            rawdata = mne.io.read_raw_edf(edf_filepath)
            data = rawdata.copy().get_data()
            raw_channels = rawdata.ch_names
            n_experiments_avail += 1
            # create montage
            order_anode = [raw_channels.index(ch) for ch in channel_anode]
            order_cathode = [raw_channels.index(ch) for ch in channel_cathode]
            # nd_anode = nd_edf[order_anode]
            # nd_cathode = nd_edf[order_cathode]
            # nd_montage = nd_anode - nd_cathode
            montage = data[order_anode] - data[order_cathode]
            n_channels.append(montage.shape[0])

            # # pick channels and create montage by mne
            # raw_edf_sel = raw_edf.copy().pick_channels(channel_names)
            # raw_montage = mne.set_bipolar_reference(
            #     raw_edf_sel.copy().load_data(),
            #     anode=channel_anode,
            #     cathode=channel_cathode,
            #     ch_name=new_channel_names
            # )
            # nd_montage = raw_montage.copy().get_data()

            # ==================================
            sfreq = rawdata.info['sfreq']
            montage = -montage
            montage = mne.filter.resample(
                montage, up=1, down=sfreq / downrate
            )  # resample
            montage = mne.filter.filter_data(
                montage,
                sfreq=sfreq, l_freq=1, h_freq=45
            ) * 1e7  # filter
            subeeg = mne.epochs.detrend(montage)  # detrend
            # # filter
            # nd_montage_filter = mne.filter.filter_data(
            #     new_nd_montage,
            #     sfreq=sfreq, l_freq=1, h_freq=45
            # ) * 1e7
            # # resample or not
            # if sfreq != downrate:
            #     nd_montage_resample = mne.filter.resample(
            #         nd_montage_filter, up=1, down=sfreq / downrate
            #     )  # resample
            #     downsample_list.append(sub_experiment)  # record downrated samples
            # else:
            #     nd_montage_resample = nd_montage_filter
            # # detrend or not
            # if detrend:
            #     subeeg = mne.epochs.detrend(nd_montage_resample)
            # else:
            #     subeeg = nd_montage_resample
            # ===============================

            # read rec file
            rec_txt = np.loadtxt(rec_filepath, delimiter=',')
            cond_label = rec_txt[:, 3] <= 3  # 6 spikes to 1/0 variable
            cond_label = cond_label.astype('int')  # change bool values to integers (1/0)
            rec_txt = np.c_[rec_txt, cond_label]  # [ch_loc, t_start, t_end, 1-6_types, 0/1_lab]
            # rec_txt[cond_label, 3] = 1
            # rec_txt[~cond_label, 3] = 0
            # rec_txt[:, [0, -1]] = rec_txt[:, [-1, 0]]  # swap the first column and the last column
            rec_txt = np.round(rec_txt, 1)  # np.round: round to float
            rec_txt = rec_txt[rec_txt[:, 1].argsort()]  # ascending sort by the second column (t_start)

            # remove 90% overlapping time periods
            # not necessary after round time to 0.1, largest overlapping = 0.9 / 1.1 < 90%
            rec_time = rec_txt[:, 1:3]  # select two columns of time
            # for i in range(rec_time.shape[0] - 1):
            #     overlap_ratio = (rec_time[i, 1] - rec_time[i + 1, 0]) / (rec_time[i + 1, 1] - rec_time[i, 0])
            #     if 0.9 <= overlap_ratio < 1:
            #         count_overlap += 1
            #         rec_time[i + 1, :] = rec_time[i, :]
            #         overlap_exp.append(sub_experiment)
            # rec_txt[:, 1:3] = rec_time
            rec_time = np.unique(rec_time, axis=0)  # select unique periods

            # segment data
            for rec in range(rec_time.shape[0]):
                t_start = rec_time[rec, 0]
                t_end = rec_time[rec, 1]
                nt_start = round(t_start * nX)  # round to integer
                nt_end = round(t_end * nX)
                if nt_end - nt_start != nX:
                    time_rm_exp.append(sub_experiment)
                    continue  # the following statements will not be executed
                t0_start = nt_start - nZ
                t0_end = nt_end + nZ
                if t0_start >= 0 and t0_end <= subeeg.shape[1]:
                    eeg = subeeg[:, t0_start: t0_end]
                    if eeg.shape[1] != nX + nZ * 2:
                        time_prob_exp.append(sub_experiment)
                    cond_channel = np.where((rec_txt[:, 1] == t_start)
                                            & (rec_txt[:, 2] == t_end))
                    info = rec_txt[cond_channel, :][0]  # select info during the period
                    spike_type = np.unique(info[:, -2])  # 1-6 unique types in the 1s epoch
                    spike_type_per_patient = np.concatenate((spike_type_per_patient, spike_type))
                    if len(np.unique(info[:, -1])) > 1:  # with both IED/nonIED labels on an epoch
                        seg_num_lab.append(sub_experiment)
                    #####
                    #####
                    target1 = max(info[:, -1])  # 1 for containing IEDs
                    target0 = min(info[:, -1])  # 0 for no IEDs
                    target = np.array([target1])
                    target_list.append(target)
                    # if labeling == 'IED_others':  # 1 for containing IEDs; 0 for others
                    #     target1 = max(info[:, -1])
                    #     target0 = min(info[:, -1])
                    #     target = np.array([target1])
                    #     target_list.append(target)
                    # elif labeling == 'IED_nonIED':  # 1 for only IED events; 0 for non-IED events
                    #     target1 = max(info[:, -1])
                    #     target0 = min(info[:, -1])
                    #     if target1 != target0:
                    #         count_IED_nonIED += 1  # 1s-epoch contains both IED & non-IED events
                    #         continue  # jump out of the closest for loop
                    #     target = np.array([target1])
                    #     target_list.append(target)
                    # elif labeling == 'IEDbckg_bckg':  # 1 for IED w/ or w/o bckg; 0 for bckg only
                    #     if (4 in spike_type) or (5 in spike_type):
                    #         count_eyem_artf += 1  # 1s-epoch contains eyem or artf
                    #         continue  # jump out of the closest for loop
                    #     target1 = max(info[:, -1])
                    #     target0 = min(info[:, -1])
                    #     if target1 != target0:
                    #         count_IED_bckg += 1  # 1s-epoch contains both IED & bckg
                    #         # not skip
                    #     target = np.array([target1])
                    #     target_list.append(target)
                    # else:
                    #     raise Exception("Labeling mis-specified.")
                    #####
                    sample_id += 1
                    #####
                    # count number of channels that contain spsw/gped/pled/eyem/artf/bckg
                    if 1 in spike_type:
                        n_ch_spsw.append(info[np.where(info[:, -2] == 1), :][0].shape[0])
                    if 2 in spike_type:
                        n_ch_gped.append(info[np.where(info[:, -2] == 2), :][0].shape[0])
                    if 3 in spike_type:
                        n_ch_pled.append(info[np.where(info[:, -2] == 3), :][0].shape[0])
                    if 4 in spike_type:
                        n_ch_eyem.append(info[np.where(info[:, -2] == 4), :][0].shape[0])
                    if 5 in spike_type:
                        n_ch_artf.append(info[np.where(info[:, -2] == 5), :][0].shape[0])
                    if 6 in spike_type:
                        n_ch_bckg.append(info[np.where(info[:, -2] == 6), :][0].shape[0])
                    #####
                    #####
                    # sample_filename = sub_experiment + '_' + str(rec) + '.npz'
                    sample_filename = str(sample_id) + '.npz'
                    res = np.c_[[str(sample_id)] * info.shape[0], info[:, [0, 3]]]
                    table = np.r_[table, res]
                    info = info[:, [0, -2]]  # columns: [ch_loc, 1-6 types]
                    time_start_end = np.array([t_start, t_end])
                    filename = np.array([sub_experiment])
                    if (not dryrun) and (save):
                        np.savez(osp.join(train_save_path, sample_filename),
                                 eeg=eeg, target=target, info=info,
                                 time=time_start_end, filename=filename)

                    # count ch numbers if all channels have annotations
                    # if info[np.where(info[:, -2] == 1), :][0].shape[0] == 22:
                    #     ch22_spsw.append(sub_experiment + ' ' + str(sample_id))
                    # if info[np.where(info[:, -2] == 2), :][0].shape[0] == 22:
                    #     ch22_gped.append(sub_experiment + ' ' + str(sample_id))
                    # if info[np.where(info[:, -2] == 3), :][0].shape[0] == 22:
                    #     ch22_pled.append(sub_experiment + ' ' + str(sample_id))
                    # if info[np.where(info[:, -2] == 4), :][0].shape[0] == 22:
                    #     ch22_eyem.append(sub_experiment + ' ' + str(sample_id))
                    # if info[np.where(info[:, -2] == 5), :][0].shape[0] == 22:
                    #     ch22_artf.append(sub_experiment + ' ' + str(sample_id))
                    # if info[np.where(info[:, -2] == 6), :][0].shape[0] == 22:
                    #     ch22_bckg.append(sub_experiment + ' ' + str(sample_id))

        except Exception as error:
            # print("An error occurred:", error)
            # print(edf_filepath)
            error_edf.append(sub_experiment)
            pass  # the codes will not be paused by the error
    spike_type_per_patient = np.unique(spike_type_per_patient)
    # count number of channels that contain spsw/gped/pled/eyem/artf/bckg
    if 1 in spike_type_per_patient:
        n_spsw += 1
    if 2 in spike_type_per_patient:
        n_gped += 1
    if 3 in spike_type_per_patient:
        n_pled += 1
    if 4 in spike_type_per_patient:
        n_eyem += 1
    if 5 in spike_type_per_patient:
        n_artf += 1
    if 6 in spike_type_per_patient:
        n_bckg += 1

# np.save(train_save_path + '_seg_ch_type.npy', table)

# save nsamples of spsw/gped/pled/eyem/artf/bckg description

pd_spsw = pd.DataFrame({'spsw': n_ch_spsw}).describe()
pd_gped = pd.DataFrame({'gped': n_ch_gped}).describe()
pd_pled = pd.DataFrame({'pled': n_ch_pled}).describe()
pd_eyem = pd.DataFrame({'eyem': n_ch_eyem}).describe()
pd_artf = pd.DataFrame({'artf': n_ch_artf}).describe()
pd_bckg = pd.DataFrame({'bckg': n_ch_bckg}).describe()

pd_merge = pd_spsw.merge(
    pd_gped, left_index=True, right_index=True
).merge(
    pd_pled, left_index=True, right_index=True
).merge(
    pd_eyem, left_index=True, right_index=True
).merge(
    pd_artf, left_index=True, right_index=True
).merge(
    pd_bckg, left_index=True, right_index=True
)

ratio1 = sum(target_list) / len(target_list)
ratio0 = 1 - ratio1

#################
# ================
#################

print("Channel descriptioin:")
print(pd.DataFrame(n_channels).describe())

if not dryrun:
    # Saving the reference of the standard output
    original_stdout = sys.stdout

    with open(osp.join(train_save_root, data_type + '_info.txt'), 'w') as f:
        sys.stdout = f
        # This message will be written to a file.

        print("Number of subjects (patients): " + str(n_subjects))
        print("Number of experiments of all subjects: " + str(n_sub_experiments))
        print("Number of experiments available: " + str(n_experiments_avail))
        print('\n')

        print("Number fo patients that contain 6 classes of labels:")
        print({"spsw": n_spsw,
               "gped": n_gped,
               "pled": n_pled,
               "eyem": n_eyem,
               "artf": n_artf,
               "bckg": n_bckg})

        print("Problematic edf files:")
        print(error_edf)
        print("Total number of overlapping: " + str(count_overlap))
        # print(overlap_exp)
        print("Total number of unique overlapping: " + str(len(set(overlap_exp))))
        print(set(overlap_exp))
        print("Removed experiments:")
        print(time_rm_exp)
        print("Time problematic experiments:")
        print(time_prob_exp)
        print("Downsampled segments:")  # should be empty
        print(downsample_list)
        print("Number of segments with both IED/nonIED labels: " + str(len(seg_num_lab)))
        print('\n')
        print('TARGET RATIO 0 / 1 = ' + str(ratio0) + ' : ' + str(ratio1))
        print('\n\n\n')

        print("CHANNEL ANNOTATION DESCRIPTION of samples that contain 1-6 types of annotations:")
        print(pd_merge)
        print('\n')

        print("LABELING TYPE: " + labeling)
        # print("Detrend: " + str(detrend))
        print('Number of samples generated (for ' + data_type + ' data):')
        if data_type == 'train':
            print(str(sample_id))
        if data_type == 'eval':
            print(str(sample_id - 10000))

        # if labeling == 'IED_nonIED':
        #     print('Numer of samples (contain both IED & non-IED events):')
        #     print(str(count_IED_nonIED))
        # if labeling == 'IEDbckg_bckg':
        #     print('Numer of samples (contain eyem or artf):')
        #     print(str(count_eyem_artf))
        #     print('Number of samples (contain both IED & bckg):')
        #     print(str(count_IED_bckg))

        # Reset the standard output
        sys.stdout = original_stdout

print('This message will be written to the screen.')

# print('Samples with labels in 22 channels - spsw: ' + str(len(ch22_spsw)))
# print('Samples with labels in 22 channels - gped: ' + str(len(ch22_gped)))
# print('Samples with labels in 22 channels - pled: ' + str(len(ch22_pled)))
# print('Samples with labels in 22 channels - eyem: ' + str(len(ch22_eyem)))
# print('Samples with labels in 22 channels - artf: ' + str(len(ch22_artf)))
# print('Samples with labels in 22 channels - bckg: ' + str(len(ch22_bckg)))
#
# print(ch22_spsw)
# print(ch22_gped)
# print(ch22_pled)

end = time.time()

print('The execution time of the program is ' + str(round((end - start) / 60)) + ' minute(s).')
