import argparse
import pathlib
import torch
import numpy as np
from tqdm import tqdm
import os

import commons
import utils


def gen_mel(filename, stft, hps):
    audio, sampling_rate = utils.load_wav_to_torch(filename)
    if sampling_rate != stft.sampling_rate:
        raise ValueError("{} {} SR doesn't match target {} SR".format(sampling_rate, stft.sampling_rate))
    audio_norm = audio / hps.max_wav_value
    audio_norm = audio_norm.unsqueeze(0)
    melspec = stft.mel_spectrogram(audio_norm)
    melspec = torch.squeeze(melspec, 0)
    
    return melspec.numpy()


def standardize_dir(args):
    if os.path.exists(args.ignore_file):
        with open(args.ignore_file, "r") as f:
            ignore_list = set(f.read().split('\n')[:-1])
    else:
        with open(args.ignore_file, "w") as f:
            ignore_list = set([])
            
    if not os.path.exists(args.error_file):
        with open(args.error_file, "w") as f:
            pass
        
    hps = utils.get_hparams_from_file(args.config_file).data
    stft = commons.TacotronSTFT(
        hps.filter_length, hps.hop_length, hps.win_length,
        hps.n_mel_channels, hps.sampling_rate, hps.mel_fmin, hps.mel_fmax
    )
        
    print("Reading directory ...")
    directory = pathlib.Path(args.input_file)
    
    wav_file = []
    for path in directory.rglob("*.wav"):
        wav_file.append(str(path))
    
    for filename in tqdm(wav_file):
        if filename in ignore_list:
            continue
        try:
            melspec = gen_mel(filename, stft, hps)
            np.save(filename.replace(".wav", ".mel.npy"), melspec)
            
            with open(args.ignore_file, "a") as f:
                f.write(filename + "\n")
        except:
            with open(args.error_file, "a") as f:
                f.write(filename + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", required=True)
    parser.add_argument("--config_file", required=True)
    parser.add_argument("--ignore_file", required=True)
    parser.add_argument("--error_file", required=True)
    args = parser.parse_args()
    
    standardize_dir(args)