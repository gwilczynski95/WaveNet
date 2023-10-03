import librosa
import numpy as np
from pathlib import Path
import soundfile as sf
import torch
from tqdm import tqdm

import wavenet.config as config
from wavenet.model import WaveNet
from wavenet.data import mu_law_decode, mu_law_encode, one_hot_encode, load_audio


class Generator:
    def __init__(self, args):
        self.args = args
        self._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.wavenet = WaveNet(args.layer_size, args.stack_size, args.in_channels, args.res_channels)
        self.wavenet.load(args.model_dir, step=args.step)

        self._temperature = args.temperature
        self._seed = load_audio(args.seed, sample_rate=args.sample_rate, reshape=False)
        self._no_of_out_samples = int(args.sample_rate * args.sec)
        self._out_filename = f"{Path(args.model_dir).parent.name}_temp_{self._temperature}.wav"

        self.output = np.zeros(self._no_of_out_samples, dtype=np.float32)
        self._input_buffer = self._prepare_input_buffer(
            self._seed, self.wavenet.receptive_fields, self.wavenet.in_channels
        )

        self._input_buffer = torch.from_numpy(self._input_buffer).float().to(self._device)

    @staticmethod
    def _prepare_input_buffer(data, receptive_fields, quant_levels):
        out = data[:receptive_fields + 1]
        out /= np.max(np.abs(out))
        out = mu_law_encode(out, quant_levels)
        out = one_hot_encode(out, quant_levels)
        out = np.transpose(out, (1, 0))
        out = np.expand_dims(out, 0)
        return out

    def generate(self):
        print("Generate audio")
        for idx in tqdm(range(self._no_of_out_samples)):
            outputs = self.wavenet.generate(self._input_buffer)
            out_dist = torch.nn.functional.softmax(
                outputs, 1
            ).cpu().numpy()[0, :, 0]  # we only generate one sample, hence the 0 in the last dim

            out_dist = out_dist / self._temperature if self._temperature else out_dist

            out_sample = np.random.choice(
                self.wavenet.in_channels,
                p=out_dist
            )

            decoded_sample = mu_law_decode(out_sample, self.wavenet.in_channels)
            self.output[idx] = decoded_sample

            onehoted_sample = np.transpose(
                one_hot_encode(np.array([out_sample]), self.wavenet.in_channels),
                (1, 0)
            )
            self._input_buffer[0, :, :-1] = self._input_buffer[0, :, 1:].clone()
            self._input_buffer[0, :, -1] = torch.Tensor(onehoted_sample).float().to(self._device)[:, 0]

    def save(self):
        print("Save audio")
        sf.write(
            self._out_filename, self.output, self.args.sample_rate, "PCM_24"
        )


if __name__ == '__main__':
    _args = config.parse_args(is_training=False)

    generator = Generator(_args)
    generator.generate()
    generator.save()
