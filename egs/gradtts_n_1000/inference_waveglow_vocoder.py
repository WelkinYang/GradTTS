import matplotlib.pyplot as plt

import sys
sys.path.append('../../waveglow/')
sys.path.append('../../')
import librosa
import numpy as np
import os
import glob
import json

import torch
from text import text_to_sequence, cmudict
from text.symbols import symbols
import commons
import attentions
import modules
import models
import utils
import soundfile as sf

def save_wav(wav, path, sample_rate, norm=False):
    if norm:
        wav *= 32767 / max(0.01, np.max(np.abs(wav)))
        wavfile.write(path, sample_rate, wav.astype(np.int16))
    else:
        sf.write(path, wav, sample_rate)

# load WaveGlow
waveglow_path = '../../waveglow/waveglow_256channels_universal_v5.pt' # or change to the latest version of the pretrained WaveGlow.
waveglow = torch.load(waveglow_path)['model']
for k, m in waveglow.named_modules():
   m._non_persistent_buffers_set = set()
waveglow = waveglow.remove_weightnorm(waveglow)
_ = waveglow.cuda().eval()

# If you are using your own trained model
model_dir = sys.argv[1]
test_files_path = sys.argv[2]

hps = utils.get_hparams_from_dir(model_dir)
checkpoint_path = utils.latest_checkpoint_path(model_dir)

# If you are using a provided pretrained model
# hps = utils.get_hparams_from_file("./configs/any_config_file.json")
# checkpoint_path = "/path/to/pretrained_model"

model = models.DiffusionGenerator(
    len(symbols) + getattr(hps.data, "add_blank", False),
    enc_out_channels=hps.data.n_mel_channels,
    **hps.model).to("cuda")

utils.load_checkpoint(checkpoint_path, model)
_ = model.eval()

cmu_dict = cmudict.CMUDict(hps.data.cmudict_path)

# normalizing & type casting
def normalize_audio(x, max_wav_value=hps.data.max_wav_value):
    return np.clip((x / np.abs(x).max()) * max_wav_value, -32768, 32767).astype("int16")

print(test_files_path)
test_lines = open(test_files_path, 'r', encoding='utf-8').readlines()
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
        (y_gen_tst, *_), *_, (attn_gen, *_) = model(x_tst, x_tst_lengths, gen=True, length_scale=length_scale)
        try:
            audio = waveglow.infer(y_gen_tst.half(), sigma=.666)
        except:
            audio = waveglow.infer(y_gen_tst, sigma=.666)

        save_wav(normalize_audio(audio[0].clamp(-1,1).data.cpu().float().numpy()), os.path.join('test_outputs', file_name), sample_rate=hps.data.sampling_rate)
