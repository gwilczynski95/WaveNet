## My own WaveNet fun

This repo is based on [another one](https://github.com/golbin/WaveNet).

[The paper](https://arxiv.org/pdf/1609.03499.pdf) implemented in this repo (WaveNet: A Generative Model for Raw Audio).

In here I just implemented the non-conditioned WaveNet, hence I was only interested in generating some piano music.

## Dataset
For the training I used the MAPS dataset. Then I splitted audio files into 2s chunks with some stride (for better
Torch Generator performance). 

## Train
To train the model basically run the `train.py` script. Remember to set some arguments (listed in `wavenet/config.py`).
Arguments should be self-explanatory. If not - contact me.

## Generate
To generate audio run `generate.py` script, setting arguments listed in `wavenet/config.py`. REMEMBER! You have to have
an already trained model

## Reuse my work
You can download my trained model from 
[here](https://drive.google.com/file/d/1XYsn3uBzOJgvWTpMgKgAVPw1o3Ndqeox). 
This is the whole model directory, so for the `model_dir` argument you'll have to pass `your/path/to/model_dir/model` 
(this is the directory with .

### Note

If you feel this README lacks something - let me know, I'll fix this. Or if you have any idea on how to improve this
project, once again, let me know ;).