from pathlib import Path

import librosa
import numpy as np
import torch.utils.data as data


def load_audio(filename, sample_rate=16000, trim=False, trim_frame_length=2048, reshape=True):
    audio, _ = librosa.load(filename, sr=sample_rate, mono=True)
    if reshape:
        audio = audio.reshape(-1, 1)

    if trim > 0:
        audio, _ = librosa.effects.trim(audio, frame_length=trim_frame_length)

    return audio


def one_hot_encode(data, channels=256):
    one_hot = np.zeros((data.size, channels), dtype=float)
    one_hot[np.arange(data.size), data.ravel()] = 1

    return one_hot


def one_hot_decode(data, axis=1):
    decoded = np.argmax(data, axis=axis)

    return decoded


def mu_law_encode(audio, quantization_channels=256):
    """
    Quantize waveform amplitudes.
    Reference: https://github.com/vincentherrmann/pytorch-wavenet/blob/master/audio_data.py
    """
    mu = float(quantization_channels - 1)
    quantize_space = np.linspace(-1, 1, quantization_channels)

    quantized = np.sign(audio) * np.log(1 + mu * np.abs(audio)) / np.log(mu + 1)
    quantized = np.digitize(quantized, quantize_space) - 1

    return quantized


def mu_law_decode(output, quantization_channels=256):
    """
    Recovers waveform from quantized values.
    Reference: https://github.com/vincentherrmann/pytorch-wavenet/blob/master/audio_data.py
    """
    mu = float(quantization_channels - 1)

    expanded = (output / quantization_channels) * 2. - 1
    waveform = np.sign(expanded) * (
                   np.exp(np.abs(expanded) * np.log(mu + 1)) - 1
               ) / mu

    return waveform


class MapsRandomDataset(data.Dataset):
    def __init__(self, data_dir, receptive_fields, sample_size, in_channels=256, sample_rate=16000):
        self.root_path = data_dir
        self.sample_rate = sample_rate
        self.sample_size = sample_size
        self.receptive_fields = receptive_fields
        self.in_channels = in_channels

        self.filenames = list(Path(self.root_path).glob("*ogg"))

    def __len__(self):
        return len(self.filenames)

    def _quantize(self, audio):
        audio = mu_law_encode(audio, self.in_channels)
        audio = mu_law_decode(audio, self.in_channels).astype(np.float32)
        return audio

    def __getitem__(self, index):
        raw_audio = load_audio(self.filenames[index], self.sample_rate, trim=False, reshape=False)

        max_val = np.max(np.abs(raw_audio))
        raw_audio /= max_val

        random_start = np.random.randint(0, raw_audio.shape[0] - self.sample_size - 1)

        x = raw_audio[random_start: random_start + self.sample_size]
        x = mu_law_encode(x, self.in_channels)
        x = one_hot_encode(x, self.in_channels).astype(np.float32)

        x = np.transpose(x, (1, 0))

        y = mu_law_encode(raw_audio[random_start + 1 + self.receptive_fields: random_start + self.sample_size + 1],
                          self.in_channels)

        return x, y
