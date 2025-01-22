import numpy as np
import mne
import os
import os.path as osp
import pandas as pd
import math

# # sample code
# filesub = 'DA402AQB'
# path = '/Users/fjiang1/Documents/eegdata/'
# filename = path + filesub + '.m00'
# f = open(filename, "r")
# txt = f.readlines()[0]
# data = np.loadtxt(filename, skiprows=2)
# sfreq = 1000 / float(txt.split(' ')[3].split('=')[1])
# fixed_chnames = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'Fz',
#                  'Cz', 'Pz']
# info = mne.create_info(fixed_chnames, sfreq=sfreq)
# rawdata = mne.io.RawArray(data[:, :19].transpose(), info)
# mne.export.export_raw(path + filesub + '.edf', rawdata)

# settings
path = '/Users/shirleywei/Dropbox/Data/Spike'
files = os.listdir(osp.join(path, 'newEEGdata'))
filesubs = [f[:-4] for f in files]

test_model = 'tuev'  # 'meg' or 'tuev'
calcu_ratio = False
chavg = False

if calcu_ratio:
    # calculate 1/0 ratio of MEG data
    if test_model == 'meg':
        filedir = osp.join(path, 'dsig140meg_128')
    elif test_model == 'tuev':
        filedir = '/Users/shirleywei/Dropbox/Data/Spike/tuev22eeg500_v2.0.0/train'
    else:
        raise Exception("No correct test model specified.")
    counter_all, counter_spike = 0, 0
    for f in os.listdir(filedir):
        X = np.load(osp.join(filedir, f))
        tar = X['target'][0]
        if tar == 1:
            counter_spike += 1
        counter_all += 1
    ratio = counter_spike / counter_all
    print("1/0 ratio for data: " + str(ratio))

else:
    # parameters
    if test_model == 'meg':
        downrate = 256
        nX = 64
        nZ = 32
        ratio = 0.37455223679603544  # 1/0 ratio according to MEG data
    elif test_model == 'tuev':
        downrate = 250
        nX = 250
        nZ = 125
        ratio = 0.47083820248592373
    else:
        raise Exception("No correct test model specified.")
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

    # channels and montages
    fixed_chnames = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 'F8', 'T3', 'T4', 'T5', 'T6',
                     'Fz', 'Cz', 'Pz']
    # channel_anode = [
    #     'Fp1', 'F7', 'T3', 'T5',
    #     'Fp2', 'F8', 'T4', 'T6',
    #     'T3', 'C3', 'Cz', 'C4',
    #     'Fp1', 'F3', 'C3', 'P3',
    #     'Fp2', 'F4', 'C4', 'P4'
    # ]
    # channel_cathode = [
    #     'F7', 'T3', 'T5', 'O1',
    #     'F8', 'T4', 'T6', 'O2',
    #     'C3', 'Cz', 'C4', 'T4',
    #     'F3', 'C3', 'P3', 'O1',
    #     'F4', 'C4', 'P4', 'O2'
    # ]
    # new_channel_names = [
    #     'Fp1-F7', 'F7-T3', 'T3-T5', 'T5-O1',
    #     'Fp2-F8', 'F8-T4', 'T4-T6', 'T6-O2',
    #     'T3-C3', 'C3-Cz', 'Cz-C4', 'C4-T4',
    #     'Fp1-F3', 'F3-C3', 'C3-P3', 'P3-O1',
    #     'Fp2-F4', 'F4-C4', 'C4-P4', 'P4-O2'
    # ]

    # Read tiantan data .m00
    foldername = 'newEEGdata_resample' + str(downrate) + 'Hz'
    if chavg:
        foldername = foldername + '_chavg'
    for filesub in filedict.keys():
        filename = osp.join(path, 'newEEGdata', filesub + '.m00')
        if osp.exists(filename):
            f = open(filename, "r")
            txt = f.readlines()[0]
            data = np.loadtxt(filename, skiprows=2)
            nrows.append(data.shape[1])
            if data.shape[1] < 19:
                continue
            sfreq = 1000 / float(txt.split(' ')[3].split('=')[1])
            info = mne.create_info(fixed_chnames, sfreq=sfreq)
            rawdata = mne.io.RawArray(data[:, :19].transpose(), info)
            # preprocessing
            nd_data = rawdata.copy().get_data()
            nd_data = -nd_data
            nd_data = mne.filter.resample(nd_data, up=1, down=sfreq / downrate)  # resample
            nd_data = mne.filter.filter_data(nd_data, sfreq=sfreq, l_freq=1, h_freq=45)  # do not need to * 1e7
            if chavg:
                nd_data = nd_data - nd_data.mean(0)[np.newaxis, :]  # channel avg
            nd_data = mne.epochs.detrend(nd_data)
            # ---------------------------------------
            times = np.array(filedict[filesub])
            counter = 0
            msm = nd_data.shape[1]  # total number of measurements
            if len(times) != 0:  # spike time, unit: second(s)
                pos = np.round(times * downrate)
                target = np.array([1])
            else:  # normal, randomly pick pos to crop
                pos = np.random.choice(range(nt, msm, nT), size=randn, replace=False)
                target = np.array([0])
            for p in pos:
                p = round(p)
                subeeg = nd_data[:, (p - nt): (p + nt)]
                time = p / downrate
                # if len(str(counter)) == 1:
                #     num = '0' + str(counter)
                # else:
                #     num = str(counter)
                # np.savez(osp.join(path, 'newEEGdata_resample', filesub + str(counter) + '.npz'),
                #          eeg=subeeg, target=target, time=time)
                np.savez(osp.join(path, foldername, str(sampleID) + '.npz'),
                         eeg=subeeg, target=target, time=time)
                sampleID += 1
                counter += 1
                if target[0] == 1:
                    samplespike += 1
                else:
                    samplenormal += 1
                if (samplespike / sampleID) < ratio:
                    break
            if (samplespike / sampleID) < ratio:
                break

    print("Number of samples: " + str(sampleID))
    print("Number of spikes: " + str(samplespike))
    print("Number of normal: " + str(samplenormal))
    print("1/0 ratio generated: " + str(samplespike / sampleID))
    print(nrows)
