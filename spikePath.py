import os
import os.path as osp


# ===========================================
def data_path(os_sys, read_type, write_type, spike_type):
    if os_sys == 'osx':  # read and write
    
        if read_type == 'dropbox':
            read_root = '/Users/shirleywei/Dropbox/Data'
        elif read_type == 'seagate':
            read_root = '/Volumes/Seagate5T'
        elif read_type == 'wd':
            read_root = '/Volumes/WD1T'
        else:
            raise Exception("No reading data path specified.")

        if write_type == 'dropbox':
            write_root = '/Users/shirleywei/Dropbox/Data'
        elif write_type == 'seagate':
            write_root = '/Volumes/Seagate5T'
        elif write_type == 'wd':
            write_root = '/Volumes/WD1T'
        else:
            raise Exception("No writing data path specified.")

    elif os_sys == 'win':  # read and write

        if read_type == 'dropbox':
            read_root = osp.join('C:', os.sep, 'Users', 'FBE', 'Dropbox', 'Data')
        elif read_type == 'seagate':
            read_root = 'D:'
        elif read_type == 'wd':
            read_root = 'E:'
        else:
            raise Exception("No reading data path specified.")

        if write_type == 'dropbox':
            write_root = osp.join('C:', os.sep, 'Users', 'FBE', 'Dropbox', 'Data')
        elif write_type == 'seagate':
            write_root = 'D:'
        elif write_type == 'wd':
            write_root = 'E:'
        else:
            raise Exception("No writing data path specified.")

    else:
        raise Exception("No correct os system specified.")
    
    if spike_type == 'IED':
        read_dir = osp.join(read_root, 'Spike', 'tuev_v2.0.0', 'edf')
        write_dir = osp.join(write_root, 'Spike')
    elif spike_type == 'seizure':
        read_dir = osp.join(read_root, 'Spike', 'tusz_v2.0.3', 'edf')
        write_dir = osp.join(write_root, 'Spike', 'resample_tusz_v2.0.3', 'edf_100_200_100')
    else:
        raise Exception("No correct spike type specified.")
    
    dir_dict = {
        'read_root': read_root,
        'write_root': write_root,
        'read_dir': read_dir,
        'write_dir': write_dir
    }

    return dir_dict

def work_path(os_sys, spike_type, label_type):
    if os_sys == 'osx':
        work_root_dir = '/Users/shirleywei/Dropbox/Projects/NestedDeepLearningModel'
        data_root_dir = '/Users/shirleywei/Dropbox/Data/Spike'
        work_dir = osp.join(work_root_dir, 'GitCodesMac')
    elif os_sys == 'win':
        work_root_dir = osp.join(
            'C:', os.sep, 'Users', 'FBE', 'Dropbox',
            'Projects', 'NestedDeepLearningModel'
        )
        data_root_dir = osp.join(
            'C:', os.sep, 'Users', 'FBE', 'Dropbox',
            'Data', 'Spike'
        )
        work_dir = osp.join(work_root_dir, 'GitCodesWin')
    elif os_sys == 'linux':
        work_root_dir = '/home/fwei/NestedDeepLearning'
        data_root_dir = '/scr/u/fwei/Spike'
        data_root_dir = '/home/fwei/NestedDeepLearning/EEGdata'
        work_dir = osp.join(work_root_dir, 'GitCodes')
    else:
        raise Exception("No correct os system specified.")
    # ========================================================
    if spike_type == 'IED':
        data_dir = osp.join(data_root_dir, label_type)  # the data have been removed on the remote server from /scr/u
    elif spike_type == 'seizure':
        if os_sys == 'linux':
            data_dir = osp.join(work_root_dir, 'EEGdata', 'edf_100_200_100', 'seiz_bckg')
        else:
            data_dir = osp.join(data_root_dir, 'resample_tusz_v2.0.3', 'edf_100_200_100', 'seiz_bckg')
    elif spike_type == 'sim':
        data_dir = osp.join(data_root_dir, 'sim22EEG500')
    else:
        raise Exception("No correct spike type specified.")
    # ========================================================
    model_dir = osp.join(work_dir, 'model')
    output_dir = osp.join(work_dir, 'output')
    result_dir = osp.join(work_root_dir, 'Results')
    dir_dict = {
        'work_root_dir': work_root_dir,
        'work_dir': work_dir,
        'result_dir': result_dir,
        'output_dir': output_dir,
        'model_dir': model_dir,
        'data_root_dir': data_root_dir,
        'data_dir': data_dir
    }
    return dir_dict
