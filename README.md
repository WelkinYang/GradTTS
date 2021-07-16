# GradTTS
## Unofficial Pytorch implementation of "Grad-TTS: A Diffusion Probabilistic Model for Text-to-Speech" ([arxiv](https://arxiv.org/abs/2105.06337))

## About this repo
This is an unofficial implementation of GradTTS. We created this project based on GlowTTS (https://github.com/jaywalnut310/glow-tts). We replace the GlowDecoder with DiffusionDecoder which follows the settings of the original paper. In addition, we also replace torch.distributed with horovod for convenience.

## Training and inference
Please go to egs/ folder, and see run.sh and inference_waveglow_vocoder.py for example use. And you should download waveglow checkpoint from [download_link](https://drive.google.com/file/d/1rpK8CzAAirq9sWZhe9nlfvxMF1dRgFbF/view) and put it into the waveglow folder before inference.

## Reference Materials
[Grad-TTS: A Diffusion Probabilistic Model for Text-to-Speech](https://arxiv.org/abs/2105.06337)

[Score-Based Generative Modeling through Stochastic Differential Equations](https://openreview.net/forum?id=PxTIG12RRHS)

[score_sde_pytorch](https://github.com/yang-song/score_sde_pytorch)

## Authors
Heyang Xue(https://github.com/WelkinYang) and Qicong Xie(https://github.com/QicongXie)




