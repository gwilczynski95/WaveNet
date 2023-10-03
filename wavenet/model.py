from pathlib import Path

import torch

from wavenet.blocks import WaveNetModel


class WaveNet:
    def __init__(self, layer_size, stack_size, in_channels, res_channels, lr=0.002, decay=0):

        self.net = WaveNetModel(layer_size, stack_size, in_channels, res_channels)

        self.in_channels = in_channels
        self.receptive_fields = self.net.receptive_fields

        self.lr = lr
        self.loss = self._loss()
        self.optimizer = self._optimizer()

        self.scheduler = None
        if decay:
            self.scheduler = torch.optim.lr_scheduler.ExponentialLR(
                self.optimizer, gamma=decay, last_epoch=-1
            )

        self._prepare_for_gpu()

    @staticmethod
    def _loss():
        loss = torch.nn.CrossEntropyLoss()

        if torch.cuda.is_available():
            loss = loss.cuda()

        return loss

    def _optimizer(self):
        return torch.optim.Adam(self.net.parameters(), lr=self.lr)

    def _prepare_for_gpu(self):
        if torch.cuda.device_count() > 1:
            print(f"{torch.cuda.device_count()} GPUs are detected.")
            self.net = torch.nn.DataParallel(self.net)

        if torch.cuda.is_available():
            self.net.cuda()

    def train(self, inputs, targets, get_out=False):
        """
        Train 1 time
        :param inputs: Tensor[batch, timestep, channels]
        :param targets: Torch tensor [batch, timestep, channels]
        :param get_out: Bool, should the function return model's output
        :return: float loss, optionally a Tensor[batch, timestep, channels]
        """
        outputs = self.net(inputs)

        loss = self.loss(outputs, targets.long())

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if get_out:
            return loss.data, outputs
        else:
            return loss.data

    def eval(self, inputs, targets, get_out=False):
        outputs = self.generate(inputs)
        loss = self.loss(outputs,
                         targets.long())
        if get_out:
            return loss, outputs
        else:
            return loss

    def generate(self, inputs):
        """
        Generate 1 time
        :param inputs: Tensor[batch, timestep, channels]
        :return: Tensor[batch, timestep, channels]
        """
        with torch.no_grad():
            outputs = self.net(inputs)

        return outputs

    @staticmethod
    def get_model_path(model_dir, step=0):
        basename = "wavenet"

        if step:
            return Path(model_dir, f"{basename}_{step}.pkl")
        else:
            return Path(model_dir, f"{basename}.pkl")

    def load(self, model_dir, step=-1):
        """
        Load pre-trained model
        :param model_dir:
        :param step:
        :return:
        """
        if step == -1:
            all_filenames = list(Path(model_dir).glob("*"))
            step = max(list(map(
                self._model_step, all_filenames
            )))

        print(f"Loading model from {model_dir}")

        model_path = self.get_model_path(model_dir, step)

        checkpoint = torch.load(model_path)

        self.net.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        if self.scheduler:
            self.scheduler.load_state_dict(checkpoint["scheduler"])
        return step

    @staticmethod
    def _model_step(filename):
        stem = filename.stem
        try:
            ckpt = int(stem.split("_")[1])
        except IndexError:
            ckpt = 0
        return ckpt

    def save(self, model_dir, step=0):
        print(f"Saving model into {model_dir}")

        model_path = self.get_model_path(model_dir, step)

        checkpoint = {
            "epoch": step,
            "model": self.net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict() if self.scheduler else None

        }

        torch.save(
            checkpoint,
            model_path
        )
