import numpy as np
import mne
import matplotlib.pyplot as plt

import lab2_pt1

ELECTRODE_NAMES = ['FP1', 'FP2', 'C3', 'C4', 'P7', 'P8', 'O1', 'O2']
ELECTRODE_MONTAGE = {
    "FP1": np.array([-3.022797, 10.470795, 7.084885]),
    "FP2": np.array([2.276825, 10.519913, 7.147003]),
    "C3": np.array([-7.339218, -0.774994, 11.782791]),
    "C4": np.array([6.977783, -1.116196, 12.059814]),
    "P7": np.array([-7.177689, -5.466278, 3.646164]),
    "P8": np.array([7.306992, -5.374619, 3.843689]),
    "O1": np.array([-2.681717, -9.658279, 3.634674]),
    "O2": np.array([2.647095, -9.638092, 3.818619])
}

BAND_START = 0.1
BAND_STOP = 120


def get_eeg_as_numpy_array(data_df):
    """ Returns a numpy array of dimension (# of EEG channels) x
    (# of samples), containing only EEG channel data present in
    <data_df>. The order of the rows is in ascending numeric order
    found in the initial file ordering:

        EEG Channel 1, 2, ... (# of channels)

    data_df: a pandas dataframe, with format as defined by the return
    value of lab2_pt1.load_recording_file
    """
    eeg_data = np.zeros((lab2_pt1.NUM_CHANNELS, data_df.shape[0]))
    i = 0
    for col_name in data_df.columns.values:
        if lab2_pt1.is_eeg(col_name):
            eeg_data[i] = data_df[col_name].values
            i += 1
    return eeg_data


def construct_mne(data_df):
    """ Returns an MNE Raw object, consisting of lab2_pt1.NUM_CHANNELS
    channels of EEG data.

    data_df: a pandas dataframe, with format as defined by the return
    value of lab2_pt1.load_recording_file
    """
    eeg_data = get_eeg_as_numpy_array(data_df)
    info_eeg_mne = mne.create_info(ch_names=ELECTRODE_NAMES, sfreq=lab2_pt1.SAMPLE_RATE, ch_types='eeg')
    raw_eeg_mne = mne.io.RawArray(eeg_data, info_eeg_mne)
    dig_montage = mne.channels.make_dig_montage(ch_pos=ELECTRODE_MONTAGE)
    raw_eeg_mne.set_montage(dig_montage)
    return raw_eeg_mne


def show_psd(data_mne, fmin=0, fmax=np.inf):
    """ Plots the power spectral density of the EEG signals in
    <data_mne>, limiting the range of the horizontal axis of the plot to
    [fmin, fmax].

    data_mne: MNE Raw object
    fmin: lower end of horizontal axis range
    fmax: upper end of horizontal axis range
    """
    spectrum = data_mne.compute_psd(method="welch", fmin=fmin, fmax=fmax)
    spectrum.plot(show=True)


def filter_band_pass(data_mne):
    """ Mutates data_mne, applying a band-pass filter
    with band defined by BAND_START and BAND_STOP, where
    BAND_START < BAND_STOP.

    data_mne: MNE Raw object
    """
    data_mne.filter(BAND_START, BAND_STOP)


def filter_notch_60(data_mne):
    """ Mutates data_mne, applying a notch filter
    to remove 60 Hz electrical noise

    data_mne: MNE Raw object
    """
    data_mne.notch_filter(60)


if __name__ == "__main__":
    data_df = lab2_pt1.load_recording_file("sample_data.txt")
    data_mne = construct_mne(data_df)
    show_psd(data_mne)
    
    data_mne_bp = construct_mne(data_df)
    filter_band_pass(data_mne_bp)
    show_psd(data_mne_bp)
    
    data_mne_notch = construct_mne(data_df)
    filter_band_pass(data_mne_notch)
    filter_notch_60(data_mne_notch)
    show_psd(data_mne_notch)
    
    plt.show()
    # TODO: add code here to show power spectral density plots