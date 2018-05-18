import numpy as np
import sys
sys.path.append('../')
import load_data
from shutil import copy2
import json
import subprocess

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
    wav_dict    = {}
    with open(file_path) as fp:
        lines = fp.readlines()
        for line in lines:
            parts   = line.split()
            if parts[0] in keys_full_lines.values():
                wav_dict[parts[0]] = parts[1]

    # Make sure that the amount of wav paths is equal to
    assert len(keys_full_lines.keys()) == len(wav_dict.keys())

    # Copy wav file for each speaker to a separate folder
    target_folder   = '/home/mark/Downloads/places_validation/'
    file_path       = '/home/mark/Downloads/placesaudio_distro_part_1/placesaudio_distro_part_1/'
    count_not_found = 0
    files_not_found = {}
    files_found     = {}
    for key, value in wav_dict.items():
        # For some reason, some wav files aren't in the zip file
        try:
            copy2(file_path + value, target_folder + value.split('/')[-1])
        except FileNotFoundError as e:
            count_not_found += 1
            files_not_found[key] = value

        files_found[key.split('-')[0]] = value.split('/')[-1]

    # 34 wav files weren't found
    # print('Amount of files not found: {}'.format(count_not_found))

    fill_wav_manually(files_found, files_not_found)

def fill_wav_manually(files_found, files_not_found):
    """
    For not found files, pick another wav file by hand (programmatically no success)
    """

    # A single dictionary in order to easily copy the files to the other folder
    manually_found = {}
    for key, value in files_not_found.items():
        manually_found[key.split('-')[0]] = ''

    len_before_assignment = len(manually_found.keys())

    manually_found['A15PYQVCGSES3G'] = 'wavs/28/utterance_346059.wav'
    manually_found['A1AEFJMOP7OZYS'] = 'wavs/64/utterance_62827.wav'
    manually_found['A1A9OMCBJQL15S'] = 'wavs/430/utterance_208307.wav'
    manually_found['A1C2T79XTDHE39'] = 'wavs/4/utterance_168895.wav'

    manually_found['A15SDWY3P1WQX6'] = 'wavs/36/utterance_41826.wav'
    manually_found['A1APC1HPGOLX2F'] = 'wavs/138/utterance_157534.wav'
    manually_found['A1DEJWBNA5OUX1'] = 'wavs/150/utterance_40840.wav'
    manually_found['A1JBLFP5IB5UF8'] = 'wavs/48/utterance_326238.wav'

    manually_found['A1J7X0XSDTJGGP'] = 'wavs/35/utterance_374031.wav'
    manually_found['A1GGWQQFBNCUEV'] = 'wavs/246/utterance_296199.wav'
    manually_found['A191H6ZEZ9I7M3'] = 'wavs/148/utterance_381212.wav'
    # Only found a single entry in utt2wav but the folder isn't available in /wavs
    manually_found['A1FZB94LK9HWBM'] = ''

    manually_found['A1EPEMSYY5Q6GF'] = 'wavs/443/utterance_233011.wav'
    manually_found['A1C5S8FZ0UGYZX'] = 'wavs/28/utterance_240707.wav'
    manually_found['A14WOX09KHZYI8'] = 'wavs/437/utterance_202569.wav'
    manually_found['A1IQL2UN4FAMF1'] = 'wavs/406/utterance_273962.wav'

    manually_found['A1J2SQGWOID8SZ'] = 'wavs/1/utterance_162028.wav'
    manually_found['A1CJM3ULFBWN1E'] = 'wavs/232/utterance_251723.wav'
    manually_found['A1HWI4N1RJGKYY'] = 'wavs/461/utterance_306142.wav'
    manually_found['A111MNQPYBOPD0'] = 'wavs/173/utterance_177556.wav'

    manually_found['A03015341AB6VCX61DKN7'] = 'wavs/52/utterance_282287.wav'
    manually_found['A17DY4MQFC4L26'] = 'wavs/103/utterance_45148.wav'
    manually_found['A1K8U4ERJCTIQ5'] = 'wavs/422/utterance_239162.wav'
    manually_found['A1E48YYO7XP92J'] = 'wavs/260/utterance_238678.wav'

    manually_found['A116P6269SII5Y'] = 'wavs/356/utterance_231366.wav'
    manually_found['A1CDWT7K9N097'] = 'wavs/150/utterance_238251.wav'
    manually_found['A1HGH370WWDHKN'] = 'wavs/321/utterance_198460.wav'
    manually_found['A11R8G0FYA2UVQ'] = 'wavs/284/utterance_229375.wav'

    manually_found['A1A6D2RDPGVX5F'] = 'wavs/272/utterance_173256.wav'
    manually_found['A17XJC8D0QB95H'] = 'wavs/253/utterance_217783.wav'
    manually_found['A1AHYAWHM0ML7H'] = 'wavs/117/utterance_143623.wav'
    manually_found['A174D7OUJ9GKZT'] = 'wavs/389/utterance_94079.wav'

    manually_found['A1E1FGA9HQUGQ7'] = 'wavs/204/utterance_318628.wav'
    manually_found['A1AF8W1Q93TODD'] = 'wavs/63/utterance_78398.wav'

    # Make sure that no mistakes are made
    len_after_assignment = len(manually_found.keys())
    assert len_before_assignment == len_after_assignment

    # Copy files to folder. This time without try-except block because files are guaranteed to be found
    for key, value in manually_found.items():
        # Don't copy speaker that cannot be found
        if key != 'A1FZB94LK9HWBM':
            copy_single_file(value)
            files_found[key] = value.split('/')[-1]

    # Create final json file which indicates which wav I used for determining gender per speaker
    with open('../data/wav.txt', 'w') as file:
        # Reverse with json.loads()
        file.write(json.dumps(files_found))

def copy_single_file(value):
    """
    Function for copying a single file
    :param value:
    :return:
    """
    file_path = '/home/mark/Downloads/placesaudio_distro_part_1/placesaudio_distro_part_1/'
    target_folder = '/home/mark/Downloads/places_validation/'
    copy2(file_path + value, target_folder + value.split('/')[-1])


def play_audio(file_name):
    """
    Play all wav tracks and determine whether it is a male or female that is talking
    :return:
    """
    with open('../data/wav.txt') as file:
        speakers = json.loads(file.read())

    result = {}
    file_folder = '/Applications/MAMP/htdocs/places_validation/'
    for speaker, wav_file in speakers.items():
        if speaker != 'A1FZB94LK9HWBM':
            print("ID of is speaker: {}".format(speaker))
            subprocess.check_call(["afplay", file_folder + wav_file])
            # Doubt: A143W5J0USUJWX
            while True:
                gender = input("Please enter gender (0 = male, 1 = female): ")
                if gender not in ['0', '1']:
                    print('Please specify 0 or 1')
                else:
                    result[speaker] = gender
                    break

    print(result)

    with open(file_name, 'w') as file:
        # Reverse with json.loads()
        file.write(json.dumps(result))



play_audio('../data/speaker_gender_second_count.txt')