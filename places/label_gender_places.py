import numpy as np
import sys
sys.path.append('../')
import load_data
from shutil import copy2
import json
import subprocess

val_conv, val_emb, val_rec, val_spk, val_spk_int, val_text, val_mfcc = load_data.dataset_places()

## Constants
FILE_PATH_PLACES_ROOT = '/home/mark/Downloads/placesaudio_distro_part_1/placesaudio_distro_part_1/'
FILE_PATH_UTTIDS  = FILE_PATH_PLACES_ROOT + 'lists/acl_2017_val_uttids'
FILE_PATH_UTT2WAV = FILE_PATH_PLACES_ROOT + 'metadata/utt2wav'
TARGET_FOLDER     = '/home/mark/Downloads/places_validation/'

def labels_first_count():
    """
    Label gender in the Flickr8K audio caption corpus, Male = 0, Female = 1.
    Note: this numpy array is only used for the assertion in check_acl_val_file(). Val_gender is created from a
    dictionary
    :return:
    """
    return np.zeros((84,2))

def check_acl_val_file():
    """
    Function which examines the lists/acl_2017_val_uttids file
    :return:
    """
    full_lines  = []
    keys        = []
    with open(FILE_PATH_UTTIDS) as fp:
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

    wav_dict    = {}
    with open(FILE_PATH_UTT2WAV) as fp:
        lines = fp.readlines()
        for line in lines:
            parts   = line.split()
            if parts[0] in keys_full_lines.values():
                wav_dict[parts[0]] = parts[1]

    # Make sure that the amount of wav paths is equal to
    assert len(keys_full_lines.keys()) == len(wav_dict.keys())

    # Copy wav file for each speaker to a separate folder
    count_not_found = 0
    files_not_found = {}
    files_found     = {}
    for key, value in wav_dict.items():
        # For some reason, some wav files aren't in the zip file
        try:
            copy2(FILE_PATH_PLACES_ROOT + value, TARGET_FOLDER + value.split('/')[-1])
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
    :param files_found:
    :param files_not_found:
    :return:
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
    manually_found['A1CDWT7K9N097']  = 'wavs/150/utterance_238251.wav'
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
    copy2(FILE_PATH_PLACES_ROOT + value, TARGET_FOLDER + value.split('/')[-1])


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


def check_frequency_missing_speaker():
    """
    There is a single speaker which is missing. This function checks how many times this speaker occurs in val_spk.
    :return:
    """
    speaker = 'A1FZB94LK9HWBM'
    count = 0
    for val_speaker in val_spk:
        if val_speaker.split('_')[1] == speaker:
            count += 1

    return count


def create_val_gender(file_name):
    with open(file_name) as file:
        speaker_genders = json.loads(file.read())

    # Fill val_gender by checking the dictionary which contains the gender as value
    val_gender = np.zeros(val_spk.shape)
    count = 0
    for _ in val_gender:
        speaker = val_spk[count]
        if speaker.split('_')[1] != 'A1FZB94LK9HWBM':
            val_gender[count] = speaker_genders[speaker.split('_')[1]]
        count += 1

    np.save('../data/places_val_gender.npy', val_gender)

def compare_rounds(file_first_round, file_second_round):
    """
    A function to compare the results between the different rounds of classifying gender.
    :param file_first_round:
    :param file_second_round:
    :return:
    """
    with open(file_first_round) as file:
        gender_first_round = json.loads(file.read())

    with open(file_second_round) as file:
        gender_second_round = json.loads(file.read())

    # Both dictionaries should be of equal length
    assert len(gender_first_round.keys()) == len(gender_second_round.keys())

    # Check whether there is any mismatch
    for key, value in gender_second_round.items():
        if gender_first_round[key] != value:
            print('{} is not the same'.format(key))


def distribution_male_female(file_name):
    """
    Contrary to explores_places.py this function counts the amount of male and female speakers (not the number of
    entries)
    :param file_name file with gender per speaker:
    :return:
    """
    with open(file_name) as file:
        gender_speaker = json.loads(file.read())

    male   = 0
    female = 0
    for key, value in gender_speaker.items():
        if value == "0":
            male += 1
        else:
            female += 1

    return male, female


# print(distribution_male_female('../data/speaker_gender_second_count.txt'))