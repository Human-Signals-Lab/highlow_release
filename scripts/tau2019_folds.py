"""
Modified on Wed Jun 12 2024

"""

import numpy as np
import os
import pickle
import random
import librosa
import check_dirs
import pandas as pd
import pdb
from collections import OrderedDict, defaultdict
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
from collections import defaultdict

base_path = "/home/shared/TAU2019mobile/TAU-urban-acoustic-scenes-2019-mobile-development/"
city_names = ['vienna', 'barcelona', 'london', 'helsinki', 'lisbon', 'paris', 'lyon', 'stockholm', 'prague']
class_list = ["airport", "shopping_mall", "metro_station", "street_pedestrian", "public_square", "street_traffic", "tram", "bus", "metro", "park"]
le = LabelEncoder()
le.fit(class_list)
le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
print(le_name_mapping)

train_meta = pd.read_csv(base_path + 'evaluation_setup/fold1_train.csv', sep="\t") # columns are: fname, scene_label
eval_meta = pd.read_csv(base_path + 'evaluation_setup/fold1_evaluate.csv', sep="\t")

target_sr_list = [1000, 2000, 16000] 
sr = 48000
dur = 5
# Create a dictionary to store files grouped by city and id
files_by_city = defaultdict(list)
# List all files in the folder
all_files = train_meta.filename.tolist()

# Group files by city
for filename in all_files:
    _, city, file_id, _, _ = filename.split('-')
    files_by_city[city].append(filename)

# Select ~20% of files from each city to use as validation set
# all files with same locaiton ID should be on same side of split
val_files = []
for city, city_files in files_by_city.items():
    val_files_for_city = []
    num_files_to_select = int(0.20 * len(city_files))
    
    # Ensure that files with the same id are either all included or all excluded
    id_grouped_files = defaultdict(list)
    for filename in city_files:
        _, _, file_id, _, _ = filename.split('-')
        id_grouped_files[file_id].append(filename)

    # Sort dictionary by number of values per key
    sorted_id_grouped_files = dict(sorted(id_grouped_files.items(), key=lambda x: len(x[1]), reverse=False))
    for id_files in sorted_id_grouped_files.values():
        # add files from location id if label is not already in the validation set
        activities_for_city = set([item.split('-')[0] for item in val_files_for_city])
        if id_files[0].split("-")[0] not in activities_for_city:
            val_files_for_city.extend(id_files)
        # break loop when there are at least 20% of files per city in the validation set
        if len(val_files_for_city) > num_files_to_select:
            val_files.extend(val_files_for_city)
            break
# training set is all files in dev set not in validation set
train_files = list(set(all_files) - set(val_files))
train_files.sort() # sort list of files alphabetically to preserve order across different sampling rates

assert len(train_files) + len(val_files) == len(all_files)

# prepare training, validation, and evaluation splits
for target_sr in target_sr_list:
    train_wav_list, train_labels = [], []
    val_wav_list, val_labels = [], []
    eval_wav_list, eval_labels = [], []
    train_location_id, eval_location_id, val_location_id = [], [], []
    split_path = "split_full_dataset_{dur}s/sr_{sr}/".format(dur=dur, sr=target_sr)
    check_dirs.check_dir(os.path.join(base_path, split_path))

    # prepare training split
    print("preparing training sampling rate: ", target_sr)
    for i, train_entry in enumerate(train_files):   
        # load original audio to target sampling rate
        y, sr = librosa.load(os.path.join(base_path, "{f}".format(f=train_entry)), sr=target_sr)
        train_location_id.append("-".join(train_entry.split("-")[1:3]))

        # truncate audio greater than 5s
        if len(y)/sr > dur:
            y = y[0:dur*sr]
        #breakpoint()
        assert len(y) == dur*sr
        # upsample audio to 16kHz
        if target_sr != 16000:
            y = librosa.resample(y, orig_sr=target_sr, target_sr=16000)
        # encode label
        label = train_entry.split("/")[1].split("-")[0]
        enc_label = le.transform([label])

        # append labels to audio sample
        y = np.hstack([y, enc_label[0]])

        train_wav_list.append(y)
        train_labels.append(label)
        #breakpoint()     
    # save off training set
    train_wav_arr = np.asarray(train_wav_list)
    train_pkl_filename = 'seg_%ds_train_cnn14_inpt_resampled.pkl' % (dur)
    if not os.path.exists(base_path + split_path + train_pkl_filename):
        with open(base_path + split_path + train_pkl_filename, 'wb') as f:
            pickle.dump(train_wav_arr, f)
    train_wav_list, train_labels = [], []
    train_wav_arr = 0

    # prepare validation split
    print("preparing validation sampling rate: ", target_sr)
    for i, val_entry in enumerate(val_files):
        # load original audio to target sampling rate
        y, sr = librosa.load(os.path.join(base_path, "{f}".format(f=val_entry)), sr=target_sr)
        val_location_id.append("-".join(val_entry.split("-")[1:3]))

        # truncate audio greater than 5s
        if len(y)/sr > dur:
            y = y[0:dur*sr]
        assert len(y) == dur*sr

        # upsample audio to 16kHz
        if target_sr != 16000:
            y = librosa.resample(y, orig_sr=target_sr, target_sr=16000)
        # encode label
        label = val_entry.split("/")[1].split("-")[0]
        enc_label = le.transform([label])

        # append labels to audio sample
        y = np.hstack([y, enc_label[0]])

        val_wav_list.append(y)
        val_labels.append(label)

    # check that location IDs in validation set do not ovelap in training set
    for id in val_location_id:
        assert id not in train_location_id    
    # save off validation set
    val_wav_arr = np.asarray(val_wav_list)
    val_pkl_filename = 'seg_%ds_val_cnn14_inpt_resampled.pkl' % (dur)
    if not os.path.exists(base_path + split_path + val_pkl_filename):
        with open(base_path + split_path + val_pkl_filename, 'wb') as f:
            pickle.dump(val_wav_arr, f)
    val_wav_list = []

    # prepare evaluation set
    print("preparing eval sampling rate: ", target_sr)
    for i, eval_entry in eval_meta.iterrows():
        # load original audio to target sampling rate
        y, sr = librosa.load(os.path.join(base_path, "{f}".format(f=eval_entry.filename)), sr=target_sr)
        eval_location_id.append("-".join(eval_entry.filename.split("-")[1:3]))

        # truncate audio greater than 5s
        if len(y)/sr > dur:
            y = y[0:dur*sr]
        assert len(y) == dur*sr

        # upsample audio to 16kHz
        if target_sr != 16000:
            y = librosa.resample(y, orig_sr=target_sr, target_sr=16000)

        # encode label
        label = eval_entry.scene_label
        enc_label = le.transform([label])
        
        # append labels to audio sample
        y = np.hstack([y, enc_label[0]])

        eval_wav_list.append(y)
        eval_labels.append(label)
    # check that location IDs in evaluation set do not ovelap in training set
    for id in eval_location_id:
        assert id not in list(set(train_location_id + val_location_id))
    # convert list to array
    eval_wav_arr = np.asarray(eval_wav_list)

    eval_pkl_filename = 'seg_%ds_eval_cnn14_inpt_resampled.pkl' % (dur)
    if not os.path.exists(base_path + split_path + eval_pkl_filename):
        with open(base_path + split_path + eval_pkl_filename, 'wb') as f:
            pickle.dump(eval_wav_arr, f)
    eval_wav_list, eval_labels = [], []
