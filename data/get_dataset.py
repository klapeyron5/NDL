import os
import numpy as np


def download(urls, dst_path):
    """
    Downloads train.zip and val.zip
    :param urls: urls of .zip archives
    :param dst_path: where to save downloaded archives
    :return:
    """
    # done by hands
    pass


def extract(src_path, dst_path):
    """
    Extracts train.zip and val.zip
    Removes .zip archives after successful extraction
    :param src_path: where .zip archives are stored
    :param dst_path: where to save extracted
    :return:
    """
    # done by hands
    pass


PROBLEM_CLASSIFICATION = 'classification'
PROBLEM_DENOISING = 'denoising'
LABEL_CLEAN = 0
LABEL_NOISY = 1

CLEAN_DIRNAME = 'clean'
NOISY_DIRNAME = 'noisy'
TRACK_EXTENSION = '.npy'


def get_xy(path, problem=PROBLEM_CLASSIFICATION):
    """
    Returns X, Y, where X, Y depend on problem:
    problem==classification: X - track files (.npy paths); Y - label 0 (clean) or 1 (noisy)
    problem==denoising: X - noisy track files (.npy paths); Y - clean track files (.npy paths)
    :param path: directory with "clean" and "noisy" directories
    :param problem: classification or denoising
    :return: X, Y
    """
    # check data storage topology
    clean_path = os.path.join(path, CLEAN_DIRNAME)
    noisy_path = os.path.join(path, NOISY_DIRNAME)
    errmsg = 'ERROR: {} not a directory'
    assert os.path.isdir(clean_path), errmsg.format(clean_path)
    assert os.path.isdir(noisy_path), errmsg.format(noisy_path)

    speakers_clean = os.listdir(clean_path)
    speakers_noisy = os.listdir(noisy_path)
    speakers = speakers_clean.copy()
    errmsg = 'ERROR: set of speakers should be the same for clean and noisy'
    assert len(speakers_clean) == len(speakers_noisy), errmsg
    assert all([x == y for x, y in zip(speakers_clean, speakers_noisy)]), errmsg
    speakers_clean = [os.path.join(clean_path, x) for x in speakers_clean]
    speakers_noisy = [os.path.join(noisy_path, x) for x in speakers_noisy]
    errmsg = 'ERROR: some speaker from {} is not a directory'
    assert all([os.path.isdir(x) for x in speakers_clean]), errmsg.format(clean_path)
    assert all([os.path.isdir(x) for x in speakers_noisy]), errmsg.format(noisy_path)
    X = []
    Y = []
    for speaker in speakers:
        speaker_clean = os.path.join(clean_path, speaker)
        speaker_noisy = os.path.join(noisy_path, speaker)
        clean_tracks = os.listdir(speaker_clean)
        noisy_tracks = os.listdir(speaker_noisy)
        errmsg = 'ERROR: set of tracks should be the same for clean and noisy for speaker {}'
        assert len(clean_tracks) == len(noisy_tracks), errmsg.format(speaker)
        assert all([x == y for x, y in zip(clean_tracks, noisy_tracks)]), errmsg.format(speaker)
        clean_tracks = [os.path.join(speaker_clean, x) for x in os.listdir(speaker_clean)]
        noisy_tracks = [os.path.join(speaker_noisy, x) for x in os.listdir(speaker_noisy)]
        errmsg = 'ERROR: some track from {} is not a file or does not end with .npy'
        def is_track(track_filepath: str):
            return os.path.isfile(track_filepath) and track_filepath.endswith(TRACK_EXTENSION)
        assert all([is_track(x) for x in clean_tracks]), errmsg.format(speaker_clean)
        assert all([is_track(x) for x in noisy_tracks]), errmsg.format(speaker_noisy)
        if problem == PROBLEM_CLASSIFICATION:
            for clean_x, noisy_x, clean_y, noisy_y in \
                    zip(clean_tracks, noisy_tracks, [LABEL_CLEAN]*len(clean_tracks), [LABEL_NOISY]*len(noisy_tracks)):
                X.append(clean_x)
                X.append(noisy_x)
                Y.append(clean_y)
                Y.append(noisy_y)
        elif problem == PROBLEM_DENOISING:
            X.extend(noisy_tracks)
            Y.extend(clean_tracks)
        else:
            raise Exception('wrong problem')

    print_stat = True
    if print_stat:
        print('data stat:')
        print('path: {}'.format(path))
        print('number of samples: {}'.format(len(X)))

    if problem == PROBLEM_CLASSIFICATION:
        unittest_xy_classification(path, X, Y)
    elif problem == PROBLEM_DENOISING:
        pass

    X = np.array(X)
    Y = np.array(Y)
    return X, Y


def unittest_xy_classification(path, X, Y):
    assert len(X) == len(Y)
    d = {}  # {speaker: {clean/noisy: [track]}}
    CLEAN_NOISY = {CLEAN_DIRNAME, NOISY_DIRNAME}
    for x, y in zip(X, Y):
        assert os.path.isfile(x)
        assert x.endswith(TRACK_EXTENSION)
        track = os.path.basename(x)
        x_split = x.split(os.path.sep)
        clean_noisy = x_split[-3]
        speaker = x_split[-2]
        assert clean_noisy in CLEAN_NOISY, clean_noisy
        if speaker not in d.keys():
            d[speaker] = {CLEAN_DIRNAME: [], NOISY_DIRNAME: []}
        d[speaker][clean_noisy].append(track)
        if clean_noisy == CLEAN_DIRNAME:
            assert y == LABEL_CLEAN
        elif clean_noisy == NOISY_DIRNAME:
            assert y == LABEL_NOISY

    clean_path = os.path.join(path, CLEAN_DIRNAME)
    speakers = os.listdir(clean_path)
    assert set(speakers) - set(d.keys()) == set()

    for speaker, v in d.items():
        clean_tracks = d[speaker][CLEAN_DIRNAME]
        noisy_tracks = d[speaker][NOISY_DIRNAME]
        assert set(clean_tracks) - set(noisy_tracks) == set()


def get_dataset(data_path, problem):
    # --- empty plugs for pipeline understanding:
    download(urls=[], dst_path=data_path)
    extract(src_path=data_path, dst_path=data_path)
    # ---

    val_path = os.path.join(data_path, 'val')
    val_X, val_Y = get_xy(val_path, problem)
    if problem == PROBLEM_CLASSIFICATION:
        assert len(val_X) == 4000
    elif problem == PROBLEM_DENOISING:
        assert len(val_X) == 2000

    train_path = os.path.join(data_path, 'train')
    trn_X, trn_Y = get_xy(train_path, problem)
    if problem == PROBLEM_CLASSIFICATION:
        assert len(trn_X) == 24000
    elif problem == PROBLEM_DENOISING:
        assert len(trn_X) == 12000

    return val_X, val_Y, trn_X, trn_Y
