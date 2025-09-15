import os
import torch
import h5py
import librosa
import pandas as pd
from torchlibrosa.stft import Spectrogram, LogmelFilterBank
import numpy as np

import os
import re
import pandas as pd

# ---------------- CONFIG ----------------
class ConvNext_conf():
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

config = ConvNext_conf()

spectrogram_extractor = Spectrogram(
    n_fft=config.window_size, hop_length=config.hop_size, win_length=config.window_size,
    window=config.window, center=config.center, pad_mode=config.pad_mode, freeze_parameters=True
)

logmel_extractor = LogmelFilterBank(
    sr=config.sample_rate, n_fft=config.window_size, n_mels=config.mel_bins,
    fmin=config.fmin, fmax=config.fmax, ref=config.ref, amin=config.amin,
    top_db=config.top_db, freeze_parameters=True
)

def log_mel_spectrogram(x):
    x = spectrogram_extractor(x) 
    x = logmel_extractor(x).squeeze()
    return x

splits=['train','val','test']
for split in splits:

    out_dir = "Audiocaps"
    os.makedirs(out_dir, exist_ok=True)

    csv_file = f"./audiocaps/{split}.csv"
    out_file_csv = f"./Audiocaps/{split}.csv"

    df = pd.read_csv(csv_file)
    
    file_names = df["youtube_id"].unique()
    fname2fid = {
        fname: os.urandom(8).hex()   # still random, but same for same youtube_id
        for fname in file_names
    }

    rows = []
    for _, row in df.iterrows():
        fname = row["youtube_id"]
        raw_text = str(row["caption"])
        tid = str(row["audiocap_id"])   #   audiocap_id as tid

        text = raw_text.lower().strip()
        tokens = [t for t in re.split(r"\s+", text) if len(t) > 0]

        fid = fname2fid[fname]  # same fid for same youtube_id

        rows.append([tid, fid, fname, raw_text, text, tokens])

    # keep all rows, sort by fid + tid
    out_df = pd.DataFrame(
        rows, columns=["tid", "fid", "fname", "raw_text", "text", "tokens"]
    )

    out_df_sorted = out_df.sort_values(by=["fid", "tid"]).reset_index(drop=True)

    out_df_sorted.to_csv(out_file_csv, index=False)


 
    dataset_dir = f"./audiocaps/{split}" #Downloaded Dataset
      
    audio_dir = dataset_dir
  
    out_hdf5 = os.path.join(out_dir, f"{split}_audio_logmels_audiocaps.hdf5")
 
    df = pd.read_csv(out_file_csv)

    # map tid → fname
    tid2fname = dict(zip(df["tid"], df["fname"]))
    tids = df["tid"].unique().tolist()

    # ---------------- PROCESS AUDIO ----------------
    with h5py.File(out_hdf5, "w") as stream:
        for tid in tids:
            wav_path = os.path.join(audio_dir, f"{tid}.wav")   # load using tid
            if not os.path.exists(wav_path):
                print("[Missing]", wav_path)
                continue

            try:
               
                y, sr = librosa.load(wav_path, sr=config.sample_rate, mono=True)
                y = torch.tensor(y).unsqueeze(0)

                # Extract log-mel
                log_mel = log_mel_spectrogram(y)

                # Save using fname (youtube_id) as dataset key
                fname = tid2fname[tid]
                stream.create_dataset(fname, data=log_mel, dtype=np.float32)

                print("[OK]", tid, "->", fname, log_mel.shape)

            except Exception as e:
                continue

    print("Finished. Features saved to:", out_hdf5)




 
splits=['train','val','test']
for split in splits:

   

    print(" Saved processed & sorted captions to:", out_file)
