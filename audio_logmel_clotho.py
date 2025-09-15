import glob
import os
import pickle
import torch
import h5py
import librosa
import numpy as np
from torchlibrosa.stft import Spectrogram, LogmelFilterBank


class cnn_14_16k_conf():
    def __init__(self):
        self.sample_rate = 16000
        self.window_size = 512
        self.hop_size = 160
        self.mel_bins = 64
        self.fmin = 50
        self.fmax = 8000
        self.window = 'hann'
        self.center = True
        self.pad_mode = 'reflect'
        self.ref = 1.0
        self.amin = 1e-10
        self.top_db = None

class cnn_14_32k_conf():  # for all 32k sample rate models
    def __init__(self):
        self.sample_rate = 32000
        self.window_size = 1024
        self.hop_size = 320
        self.mel_bins = 64
        self.fmin = 50
        self.fmax = 14000
        self.window = 'hann'
        self.center = True
        self.pad_mode = 'reflect'
        self.ref = 1.0
        self.amin = 1e-10
        self.top_db = None

class ConvNext_conf():  # for all 32k sample rate models
    def __init__(self):
        self.sample_rate = 32000
        self.window_size = 1024
        self.hop_size = 320
        self.mel_bins = 224
        self.fmin = 50
        self.fmax = 14000
        self.window = 'hann'
        self.center = True
        self.pad_mode = 'reflect'
        self.ref = 1.0
        self.amin = 1e-10
        self.top_db = None

config = ConvNext_conf() # change the configuration according to the feature extractor you need

spectrogram_extractor = Spectrogram(n_fft=config.window_size, hop_length=config.hop_size, win_length=config.window_size, 
                                    window=config.window, center=config.center, pad_mode=config.pad_mode, freeze_parameters=True)

logmel_extractor = LogmelFilterBank(sr=config.sample_rate, n_fft=config.window_size, n_mels=config.mel_bins, fmin=config.fmin, 
                                    fmax=config.fmax, ref=config.ref, amin=config.amin, top_db=config.top_db, freeze_parameters=True)

def log_mel_spectrogram(x):
    x = spectrogram_extractor(x) 
    x = logmel_extractor(x).squeeze()
    return x


# def log_mel_spectrogram(y,
#                         sample_rate=44100,
#                         window_length_secs=0.025,
#                         hop_length_secs=0.010,
#                         num_mels=128,
#                         log_offset=0.0):
#     """Convert waveform to a log magnitude mel-frequency spectrogram.

#     :param y: 1D np.array of waveform data.
#     :param sample_rate: The sampling rate of data.
#     :param window_length_secs: Duration of each window to analyze.
#     :param hop_length_secs: Advance between successive analysis windows.
#     :param num_mels: Number of Mel bands.
#     :param fmin: Lower bound on the frequencies to be included in the mel spectrum.
#     :param fmax: The desired top edge of the highest frequency band.
#     :param log_offset: Add this to values when taking log to avoid -Infs.
#     :return:
#     """
#     window_length = int(round(sample_rate * window_length_secs))
#     hop_length = int(round(sample_rate * hop_length_secs))
#     fft_length = 2 ** int(np.ceil(np.log(window_length) / np.log(2.0)))

#     mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sample_rate, n_fft=fft_length, hop_length=hop_length,
#                                                      win_length=window_length, n_mels=num_mels)

#     return np.log(mel_spectrogram + log_offset)


# %%

global_params = {
    "dataset_dir": "./data",
    "audio_splits": ["development", "validation", "evaluation"],
    
    "logmels_dir": "data_32k_224_mels" # change this directory name for feature extraction for different sample rates
}

# Load audio info
audio_info = os.path.join(global_params["dataset_dir"], "audio_info.pkl")
with open(audio_info, "rb") as store:
    audio_fid2fname = pickle.load(store)["audio_fid2fname"]


# Create logmels folder where the logmels hdf5 files will get saved
if not os.path.exists(global_params["logmels_dir"]):
    os.makedirs(global_params["logmels_dir"], exist_ok=False)

# Extract log mel for splits
for split in global_params["audio_splits"]:

    fid2fname = audio_fid2fname[split]
    fname2fid = {fid2fname[fid]: fid for fid in fid2fname}

    audio_dir = os.path.join(global_params["dataset_dir"], split)
    audio_logmel = os.path.join(global_params["logmels_dir"], f"{split}_audio_logmels.hdf5") # change logmels_sample_rate name here to save the melspecs

    with h5py.File(audio_logmel, "w") as stream:

        for fpath in glob.glob(r"{}/*.wav".format(audio_dir)):
            try:
                fname = os.path.basename(fpath)
                fid = fname2fid[fname]

                # change the feature extractor to CNN14 or ConvNext feature extractor with their configuration
                y, sr = librosa.load(fpath, sr=32000, mono=True) # change sample rate according to the extracted feature you need
                y = torch.tensor(y).reshape(1,-1)
                # log_mel = log_mel_spectrogram(y=y, sample_rate=sr, window_length_secs=0.040, hop_length_secs=0.020,
                #                               num_mels=64, log_offset=np.spacing(1))
                
                log_mel = log_mel_spectrogram(y)

                # stream[fid] = np.vstack(log_mel).transpose()  # [Time, Mel]
                stream[fid] = np.vstack(log_mel)  # [Time, Mel]
                print(fid, fname)
            except:
                print("Error audio file:", fpath)

    print("Save", audio_logmel)
