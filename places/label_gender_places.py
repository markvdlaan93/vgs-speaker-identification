import numpy as np
import sys
sys.path.append('../')
import load_data
from shutil import copy2

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
    file_path       = '/home/mark/Downloads/placesaudio_distro_part_1/placesaudio_distro_part_1/metadata/utt2wav'
    wav_path        = '/home/mark/Downloads/placesaudio_distro_part_1/placesaudio_distro_part_1/'
    target_folder   = '/home/mark/Downloads/places_validation/'
    validation_wav  = {}
    with open(file_path) as fp:
        lines = fp.readlines()
        result = []
        for line in lines:
            line = line.split()
            tag = line[0].split('-')[0]
            # if tag in validation set than save the wav name and copy the file to another folder
            if tag in val_spk_result and tag not in validation_wav.keys():
                validation_wav[tag] = line[1]
                try:
                    copy2(wav_path + line[1], target_folder + line[1].split('/')[-1])
                except FileNotFoundError as e:
                    print(e)


    # Amount of keys should be equal to counting label
    assert labels_first_count().shape[0] == len(validation_wav.keys())





load_txt_file()