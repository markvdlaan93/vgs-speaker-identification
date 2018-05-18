import numpy as np
import sys
sys.path.append('../')
import load_data
from shutil import copy2
from os.path import isdir

val_conv, val_emb, val_rec, val_spk, val_spk_int, val_text, val_mfcc = load_data.dataset_places()

def labels_first_count():
    """
    Label gender in the Flickr8K audio caption corpus, Male = 0, Female = 1.
    :return:
    """
    return np.array([
        [  1,    0],
        [  2,    0],
        [  3,    0],
        [  4,    0],
        [  5,    0],
        [  6,    0],
        [  7,    0],
        [  8,    0],
        [  9,    0],
        [ 10,    0],
        [ 11,    0],
        [ 12,    0],
        [ 13,    0],
        [ 14,    0],
        [ 15,    0],
        [ 16,    0],
        [ 17,    0],
        [ 18,    0],
        [ 19,    0],
        [ 20,    0],
        [ 21,    0],
        [ 22,    0],
        [ 23,    0],
        [ 24,    0],
        [ 25,    0],
        [ 26,    0],
        [ 27,    0],
        [ 28,    0],
        [ 29,    0],
        [ 30,    0],
        [ 31,    0],
        [ 32,    0],
        [ 33,    0],
        [ 34,    0],
        [ 35,    0],
        [ 36,    0],
        [ 37,    0],
        [ 38,    0],
        [ 39,    0],
        [ 40,    0],
        [ 41,    0],
        [ 42,    0],
        [ 43,    0],
        [ 44,    0],
        [ 45,    0],
        [ 46,    0],
        [ 47,    0],
        [ 48,    0],
        [ 49,    0],
        [ 50,    0],
        [ 51,    0],
        [ 52,    0],
        [ 53,    0],
        [ 54,    0],
        [ 55,    0],
        [ 56,    0],
        [ 57,    0],
        [ 58,    0],
        [ 59,    0],
        [ 60,    0],
        [ 61,    0],
        [ 62,    0],
        [ 63,    0],
        [ 64,    0],
        [ 65,    0],
        [ 66,    0],
        [ 67,    0],
        [ 68,    0],
        [ 69,    0],
        [ 70,    0],
        [ 71,    0],
        [ 72,    0],
        [ 73,    0],
        [ 74,    0],
        [ 75,    0],
        [ 76,    0],
        [ 77,    0],
        [ 78,    0],
        [ 79,    0],
        [ 80,    0],
        [ 81,    0],
        [ 82,    0],
        [ 83,    0],
        [ 84,    0],
        [ 85,    0]
    ])

def check_acl_val_file():
    """
    Function which examines the lists/acl_2017_val_uttids file
    :return:
    """
    file_path   = '/home/mark/Downloads/placesaudio_distro_part_1/placesaudio_distro_part_1/lists/acl_2017_val_uttids'
    full_lines  = []
    keys        = []
    with open(file_path) as fp:
        lines = fp.readlines()
        for line in lines:
            key = line.split('-')[0]
            if key not in keys:
                full_lines.append(line.rstrip())
                keys.append(key)

    # Dictionary with structure: 'A1IFIK8J49WBER': 'A1IFIK8J49WBER-GSUN_E45F9E9AA12C1220B3510C539C6004FA'
    # Key is used to check the key with val_spk while the value is used to find a specific wav file
    keys_full_lines = dict(zip(keys, full_lines))

    # Amount of keys should be equal to counting label
    assert len(keys_full_lines.keys()) == labels_first_count().shape[0]

    file_path   = '/home/mark/Downloads/placesaudio_distro_part_1/placesaudio_distro_part_1/metadata/utt2wav'
    wav_path    = '/home/mark/Downloads/placesaudio_distro_part_1/placesaudio_distro_part_1/wavs/'
    wav_dict    = {}
    with open(file_path) as fp:
        lines = fp.readlines()
        for line in lines:
            parts   = line.split()
            # Some folders aren't in the zip for some reason. Therefore, check whether folder exists
            folder = parts[1].split('/')[1]
            if parts[0] in keys_full_lines.values():
                if isdir(wav_path + folder):
                    wav_dict[parts[0]] = parts[1]

    # Make sure that the amount of wav paths is equal to
    assert len(keys_full_lines.keys()) == len(wav_dict.keys())

    # Copy wav file for each speaker to a separate folder
    target_folder   = '/home/mark/Downloads/places_validation/'
    file_path       = '/home/mark/Downloads/placesaudio_distro_part_1/placesaudio_distro_part_1/'
    count_not_found = 0
    files_not_found = {}
    files_found        = {}
    for key, value in wav_dict.items():
        # For some reason, some wav files aren't in the zip file
        try:
            copy2(file_path + value, target_folder + value.split('/')[-1])
        except FileNotFoundError as e:
            print('File {} not found'.format(value))
            count_not_found += 1
            files_not_found[key] = value

        files_found[key.split('-')[0]] = value.split('/')[-1]

    # 34 wav files weren't found
    print('Amount of files not found: {}'.format(count_not_found))

    # fill_wav_manually(files_found, files_not_found)

def fill_wav_manually(files_found, files_not_found):
    """
    For not found files, pick another wav file by hand (programmatically no success)
    """
    # files_found['A1E1FGA9HQUGQ7'] =

def copy_single_file(value):
    """
    Function for copying a single file
    :param value:
    :return:
    """
    file_path = '/home/mark/Downloads/placesaudio_distro_part_1/placesaudio_distro_part_1/'
    target_folder = '/home/mark/Downloads/places_validation/'
    copy2(file_path + value, target_folder + value.split('/')[-1])

def load_txt_file():
    """
    Retrieve for every speaker one .wav file in order to label the gender
    :return:
    """

    # Strip off prefix 'places_'
    val_spk_result = []
    for speaker in val_spk:
        val_spk_result.append(speaker.split('_')[1])

    # Open file with utterances and check whether a certain entry belongs to the validation set
    file_path       = '/home/mark/Downloads/placesaudio_distro_part_1/placesaudio_distro_part_1/lists/acl_2017_val_uttids'
    wav_path        = '/home/mark/Downloads/placesaudio_distro_part_1/placesaudio_distro_part_1/'
    target_folder   = '/home/mark/Downloads/places_validation/'
    validation_wav  = {}
    with open(file_path) as fp:
        lines = fp.readlines()
        for line in lines:
            tag = line.split('-')[0]
            # if tag in validation set than save the wav name and copy the file to another folder
            if tag in val_spk_result and tag not in validation_wav.keys():
                validation_wav[tag] = ''


    # Amount of keys should be equal to counting label
    assert labels_first_count().shape[0] == len(validation_wav.keys())




check_acl_val_file()