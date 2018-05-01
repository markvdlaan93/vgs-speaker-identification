import numpy as np
import subprocess
import majority

def labels_first_count():
    """
    Label gender in the Flickr8K audio caption corpus, Male = 0, Female = 1.
    :return:
    """
    return np.array([
        [  1,    0],
        [  2,    0],
        [  3,    0],
        [  4,    1],
        [  5,    0],
        [  6,    0],
        [  7,    0],
        [  8,    1],
        [  9,    1],
        [ 10,    0],
        [ 11,    1],
        [ 12,    0],
        [ 13,    0],
        [ 14,    0],
        [ 15,    1],
        [ 16,    0],
        [ 17,    0],
        [ 18,    1],
        [ 19,    0],
        [ 20,    1],
        [ 21,    0],
        [ 22,    1],
        [ 23,    1],
        [ 24,    1],
        [ 25,    0],
        [ 26,    0],
        [ 27,    1],
        [ 28,    1],
        [ 29,    0],
        [ 30,    1],
        [ 31,    1],
        [ 32,    1],
        [ 33,    0],
        [ 34,    0],
        [ 35,    1],
        [ 36,    1],
        [ 37,    0],
        [ 38,    1],
        [ 39,    1],
        [ 40,    1],
        [ 41,    0],
        [ 42,    1],
        [ 43,    1],
        [ 44,    1],
        [ 45,    0],
        [ 46,    0],
        [ 47,    0],
        [ 48,    0],
        [ 49,    1],
        [ 50,    1],
        [ 51,    0],
        [ 52,    1],
        [ 53,    1],
        [ 54,    1],
        [ 55,    1],
        [ 56,    1],
        [ 57,    1],
        [ 58,    0],
        [ 59,    1],
        [ 60,    0],
        [ 61,    0],
        [ 62,    1],
        [ 63,    1],
        [ 64,    0],
        [ 65,    1],
        [ 66,    1],
        [ 67,    1],
        [ 68,    0],
        [ 69,    0],
        [ 70,    1],
        [ 71,    1],
        [ 72,    1],
        [ 73,    1],
        [ 74,    1],
        [ 75,    0],
        [ 76,    0],
        [ 77,    1],
        [ 78,    0],
        [ 79,    0],
        [ 80,    1],
        [ 81,    0],
        [ 82,    0],
        [ 83,    1],
        [ 84,    1],
        [ 85,    0],
        [ 86,    1],
        [ 87,    1],
        [ 88,    0],
        [ 89,    1],
        [ 90,    0],
        [ 91,    1],
        [ 92,    0],
        [ 93,    1],
        [ 94,    1],
        [ 95,    1],
        [ 96,    0],
        [ 97,    1],
        [ 98,    1],
        [ 99,    1],
        [100,    0],
        [101,    0],
        [102,    1],
        [103,    1],
        [104,    1],
        [105,    0],
        [106,    1],
        [107,    1],
        [108,    1],
        [109,    0],
        [110,    1],
        [111,    1],
        [112,    0],
        [113,    0],
        [114,    1],
        [115,    1],
        [116,    1],
        [117,    1],
        [118,    0],
        [119,    0],
        [120,    1],
        [121,    1],
        [122,    1],
        [123,    1],
        [124,    1],
        [125,    1],
        [126,    1],
        [127,    0],
        [128,    1],
        [129,    1],
        [130,    1],
        [131,    1],
        [132,    1],
        [133,    1],
        [134,    1],
        [135,    0],
        [136,    0],
        [137,    0],
        [138,    1],
        [139,    0],
        [140,    0],
        [141,    1],
        [142,    0],
        [143,    0],
        [144,    1],
        [145,    1],
        [146,    1],
        [147,    1],
        [148,    0],
        [149,    0],
        [150,    0],
        [151,    1],
        [152,    0],
        [153,    0],
        [154,    1],
        [155,    1],
        [156,    0],
        [157,    1],
        [158,    0],
        [159,    1],
        [160,    0],
        [161,    0],
        [162,    0],
        [163,    1],
        [164,    1],
        [165,    1],
        [166,    1],
        [167,    1],
        [168,    1],
        [169,    1],
        [170,    1],
        [171,    1],
        [172,    0],
        [173,    0],
        [174,    0],
        [175,    1],
        [176,    1],
        [177,    1],
        [178,    0],
        [179,    0],
        [180,    1],
        [181,    0],
        [182,    1],
        [183,    1]
    ])

def labels_second_count():
    """
    Label gender in the Flickr8K audio caption corpus, Male = 0, Female = 1.
    :return:
    """
    return np.array([
        [  1,    0],
        [  2,    0],
        [  3,    0],
        [  4,    1],
        [  5,    0],
        [  6,    0],
        [  7,    0],
        [  8,    1],
        [  9,    1],
        [ 10,    0],
        [ 11,    1],
        [ 12,    0],
        [ 13,    0],
        [ 14,    0],
        [ 15,    1],
        [ 16,    0],
        [ 17,    0],
        [ 18,    1],
        [ 19,    0],
        [ 20,    1],
        [ 21,    0],
        [ 22,    1],
        [ 23,    1],
        [ 24,    1],
        [ 25,    0],
        [ 26,    0],
        [ 27,    1],
        [ 28,    1],
        [ 29,    0],
        [ 30,    1],
        [ 31,    1],
        [ 32,    1],
        [ 33,    0],
        [ 34,    0],
        [ 35,    0], # Doubt
        [ 36,    1],
        [ 37,    0],
        [ 38,    1],
        [ 39,    1],
        [ 40,    1],
        [ 41,    0],
        [ 42,    1],
        [ 43,    1],
        [ 44,    1],
        [ 45,    1],
        [ 46,    0],
        [ 47,    0],
        [ 48,    0],
        [ 49,    1],
        [ 50,    1],
        [ 51,    0],
        [ 52,    1], # Doubt
        [ 53,    1],
        [ 54,    1],
        [ 55,    1],
        [ 56,    1],
        [ 57,    1],
        [ 58,    0],
        [ 59,    1],
        [ 60,    0],
        [ 61,    0],
        [ 62,    1],
        [ 63,    1],
        [ 64,    0],
        [ 65,    1],
        [ 66,    1],
        [ 67,    1],
        [ 68,    0],
        [ 69,    0],
        [ 70,    1],
        [ 71,    1],
        [ 72,    1],
        [ 73,    1],
        [ 74,    1],
        [ 75,    0],
        [ 76,    0],
        [ 77,    1], # Doubt
        [ 78,    0],
        [ 79,    0],
        [ 80,    1],
        [ 81,    0],
        [ 82,    0],
        [ 83,    0],
        [ 84,    1],
        [ 85,    0],
        [ 86,    1],
        [ 87,    1],
        [ 88,    0],
        [ 89,    1],
        [ 90,    0],
        [ 91,    1],
        [ 92,    0],
        [ 93,    1],
        [ 94,    1], # Doubt
        [ 95,    1],
        [ 96,    0],
        [ 97,    1],
        [ 98,    1],
        [ 99,    1],
        [100,    0],
        [101,    0],
        [102,    0], # Doubt
        [103,    0],
        [104,    1],
        [105,    0],
        [106,    1],
        [107,    1],
        [108,    1],
        [109,    0],
        [110,    1],
        [111,    1],
        [112,    0],
        [113,    0],
        [114,    1],
        [115,    1],
        [116,    1],
        [117,    1],
        [118,    0],
        [119,    0],
        [120,    1],
        [121,    1],
        [122,    1],
        [123,    1],
        [124,    1],
        [125,    1],
        [126,    1],
        [127,    0],
        [128,    0],
        [129,    1],
        [130,    1],
        [131,    1],
        [132,    0],
        [133,    1],
        [134,    1],
        [135,    0],
        [136,    0],
        [137,    0],
        [138,    1],
        [139,    0],
        [140,    0],
        [141,    1],
        [142,    0],
        [143,    0],
        [144,    1],
        [145,    1],
        [146,    1],
        [147,    0], # Doubt
        [148,    0],
        [149,    0],
        [150,    0],
        [151,    1],
        [152,    0],
        [153,    0],
        [154,    1],
        [155,    1],
        [156,    0],
        [157,    1],
        [158,    0],
        [159,    1],
        [160,    0],
        [161,    0],
        [162,    0],
        [163,    1],
        [164,    1],
        [165,    0], # Doubt
        [166,    1],
        [167,    1],
        [168,    1],
        [169,    1],
        [170,    1],
        [171,    1],
        [172,    0],
        [173,    0],
        [174,    0], # Doubt
        [175,    1],
        [176,    1],
        [177,    0],
        [178,    0],
        [179,    0],
        [180,    1],
        [181,    0], # Doubt
        [182,    1],
        [183,    1]
    ])

def labels_final():
    """
    Label gender in the Flickr8K audio caption corpus, Male = 0, Female = 1.
    :return:
    """
    return np.array([
        [  1,    0],
        [  2,    0],
        [  3,    0],
        [  4,    1],
        [  5,    0],
        [  6,    0],
        [  7,    0],
        [  8,    1],
        [  9,    1],
        [ 10,    0],
        [ 11,    1],
        [ 12,    0],
        [ 13,    0],
        [ 14,    0],
        [ 15,    1],
        [ 16,    0],
        [ 17,    0],
        [ 18,    1],
        [ 19,    0],
        [ 20,    1],
        [ 21,    0],
        [ 22,    1],
        [ 23,    1],
        [ 24,    1],
        [ 25,    0],
        [ 26,    0],
        [ 27,    1],
        [ 28,    1],
        [ 29,    0],
        [ 30,    1],
        [ 31,    1],
        [ 32,    1],
        [ 33,    0],
        [ 34,    0],
        [ 35,    0],
        [ 36,    1],
        [ 37,    0],
        [ 38,    1],
        [ 39,    1],
        [ 40,    1],
        [ 41,    0],
        [ 42,    1],
        [ 43,    1],
        [ 44,    1],
        [ 45,    1],
        [ 46,    0],
        [ 47,    0],
        [ 48,    0],
        [ 49,    1],
        [ 50,    1],
        [ 51,    0],
        [ 52,    1],
        [ 53,    1],
        [ 54,    1],
        [ 55,    1],
        [ 56,    1],
        [ 57,    1],
        [ 58,    0],
        [ 59,    1],
        [ 60,    0],
        [ 61,    0],
        [ 62,    1],
        [ 63,    1],
        [ 64,    0],
        [ 65,    1],
        [ 66,    1],
        [ 67,    1],
        [ 68,    0],
        [ 69,    0],
        [ 70,    1],
        [ 71,    1],
        [ 72,    1],
        [ 73,    1],
        [ 74,    1],
        [ 75,    0],
        [ 76,    0],
        [ 77,    1],
        [ 78,    0],
        [ 79,    0],
        [ 80,    1],
        [ 81,    0],
        [ 82,    0],
        [ 83,    0],
        [ 84,    1],
        [ 85,    0],
        [ 86,    1],
        [ 87,    1],
        [ 88,    0],
        [ 89,    1],
        [ 90,    0],
        [ 91,    1],
        [ 92,    0],
        [ 93,    1],
        [ 94,    1],
        [ 95,    1],
        [ 96,    0],
        [ 97,    1],
        [ 98,    1],
        [ 99,    1],
        [100,    0],
        [101,    0],
        [102,    1],
        [103,    0],
        [104,    1],
        [105,    0],
        [106,    1],
        [107,    1],
        [108,    1],
        [109,    0],
        [110,    1],
        [111,    1],
        [112,    0],
        [113,    0],
        [114,    1],
        [115,    1],
        [116,    1],
        [117,    1],
        [118,    0],
        [119,    0],
        [120,    1],
        [121,    1],
        [122,    1],
        [123,    1],
        [124,    1],
        [125,    1],
        [126,    1],
        [127,    0],
        [128,    1],
        [129,    1],
        [130,    1],
        [131,    1],
        [132,    0],
        [133,    1],
        [134,    1],
        [135,    0],
        [136,    0],
        [137,    0],
        [138,    1],
        [139,    0],
        [140,    0],
        [141,    1],
        [142,    0],
        [143,    0],
        [144,    1],
        [145,    1],
        [146,    1],
        [147,    0],
        [148,    0],
        [149,    0],
        [150,    0],
        [151,    1],
        [152,    0],
        [153,    0],
        [154,    1],
        [155,    1],
        [156,    0],
        [157,    1],
        [158,    0],
        [159,    1],
        [160,    0],
        [161,    0],
        [162,    0],
        [163,    1],
        [164,    1],
        [165,    1],
        [166,    1],
        [167,    1],
        [168,    1],
        [169,    1],
        [170,    1],
        [171,    1],
        [172,    0],
        [173,    0],
        [174,    0],
        [175,    1],
        [176,    1],
        [177,    1],
        [178,    0],
        [179,    0],
        [180,    1],
        [181,    0],
        [182,    1],
        [183,    1]
    ])

def compare_rounds():
    """
    35 is not the same
    45 is not the same
    83 is not the same
    102 is not the same
    103 is not the same
    128 is not the same
    132 is not the same
    147 is not the same
    165 is not the same
    177 is not the same
    :return: ndarray
    """
    result_round_1 = labels_first_count()
    result_round_2 = labels_second_count()
    for i in range(result_round_1.shape[0]):
        if result_round_1[i][1] != result_round_2[i][1]:
            print('{} is not the same'.format(i+1))

def filter_speakers(counts):
    """
    Speaker's that should be removed: 121, 136, 159, 169, 172, 174, 175
    :param counts:
    :return:
    """
    result = labels_final()
    indices = []
    for speaker in result:
        if speaker[0] not in counts[:, 0]:
            indices.append(speaker[0]-1)
    return np.delete(result, indices, 0)

def audio_speaker(file):
    with open(file) as fp:
        lines = fp.readlines()
        result = []
        ids = []
        for line in lines:
            words = line.split()
            if words[1] not in ids:
                result.append(words)
                ids.append(words[1])

        return np.array(result)

def play_audio(audio_speakers):
    for speaker in audio_speakers:
        print("ID of is speaker: {}".format(speaker[1]))
        subprocess.check_call(["afplay", '/Applications/MAMP/htdocs/flickr_audio/wavs/' + speaker[0]])

def create_y_train(val_spk):
    """
    Creating the labels for y_train by check if speaker ID is female (== 1) or a male (== 0)
    :param val_spk:
    :return:
    """
    val_spk = np.array(val_spk.astype(int))
    val_gender = np.zeros(val_spk.shape)
    counts, _ = majority.majority(val_spk)
    labels = filter_speakers(counts)

    # Turn into dictionary in order to easily access indices
    labels_dict = {}
    for label in labels:
        labels_dict[label[0]] = label[1]

    # Fill gender array based on gender dictionary
    for i in range(val_spk.shape[0]):
        id = val_spk[i]
        gender = labels_dict[id]
        val_gender[i] = gender

    # Extra check in order to make sure that everything went alright
    female = 0
    for label in val_gender:
        female += label

    # Make sure the right gender is assigned to the right speaker
    male_count, female_count = count_occurences_male_female(val_spk)
    assert female == female_count and val_gender.shape[0] - female == male_count

    return val_gender

def count_occurences_male_female(val_spk):
    labels_gender = labels_final()

    # Count males and females in labels_gender final count (79 males, 104 males)
    male = 0
    female = 0
    for label in labels_gender:
        if not label[1]:
            male += 1
        else:
            female += 1

    counts, _ = majority.majority(val_spk)

    # Turn counts into dictionary
    counts_dict = {}
    for count in counts:
        counts_dict[count[0]] = count[1]

    # Count occurrences in validation set based on the counts of occurrences (1941 speech fragments with males,
    # 3059 speech fragments with females)
    male = 0
    female = 0
    for label in labels_gender:
        if label[0] in counts_dict:
            count = counts_dict[label[0]]
            # Comment out to see detailed view of how the numbers are calculated
            # print("For {} the count is {} with gender {}".format(label[0], count, label[1]))
            if not label[1]:
                male += count
            else:
                female += count

    return male, female

# audio_speakers = audio_speaker('/Applications/MAMP/htdocs/flickr_audio/wav2spk.txt')
# play_audio(audio_speakers)

# compare_rounds()
