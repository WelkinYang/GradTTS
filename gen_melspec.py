import matplotlib.pyplot as plt

import librosa
import numpy as np
import os
import glob
import json
import argparse

import torch
from text import text_to_sequence, cmudict
from text.symbols import symbols
import commons
import attentions
import modules
import models
import utils
import soundfile as sf
        
        
def main(args):
    hps = utils.get_hparams_from_dir(args.model_dir)
    checkpoint_path = utils.latest_checkpoint_path(args.model_dir)
    
    model = models.DiffusionGenerator(
    n_vocab=len(symbols) + getattr(hps.data, "add_blank", False),
    enc_out_channels=hps.data.n_mel_channels,
    **hps.model).cuda()

    utils.load_checkpoint(checkpoint_path, model)
    _ = model.eval()

    cmu_dict = cmudict.CMUDict(hps.data.cmudict_path)
    
    test_lines = open(args.test_files_path, 'r', encoding='utf-8').readlines()
    for line in test_lines:
        file_name = os.path.basename(line.strip().split('|')[0])
        print(file_name)

        tst_stn = line.strip().split('|')[-1]

        if getattr(hps.data, "add_blank", False):
            text_norm = text_to_sequence(tst_stn.strip(), ['english_cleaners'], cmu_dict)
            text_norm = commons.intersperse(text_norm, len(symbols))
        else: # If not using "add_blank" option during training, adding spaces at the beginning and the end of utterance improves quality
            tst_stn = " " + tst_stn.strip() + " "
            text_norm = text_to_sequence(tst_stn.strip(), ['english_cleaners'], cmu_dict)
        sequence = np.array(text_norm)[None, :]
        print("".join([symbols[c] if c < len(symbols) else "<BNK>" for c in sequence[0]]))
        x_tst = torch.autograd.Variable(torch.from_numpy(sequence)).cuda().long()
        x_tst_lengths = torch.tensor([x_tst.shape[1]]).cuda()


        with torch.no_grad():
            length_scale = 1.0
            (y_gen_tst, *_), *_, (_, *_) = model(x_tst, x_tst_lengths, gen=True, length_scale=length_scale)
            melspec = y_gen_tst.cpu().float().numpy()
            np.save(f"{args.output_dir}/{file_name}.mel.npy", melspec)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_files_path", type=str, required=True)
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)