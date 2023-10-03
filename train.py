from pathlib import Path
import pickle

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from wavenet import config
from wavenet.data import MapsRandomDataset
from wavenet.model import WaveNet


class Trainer:
    def __init__(self, args):
        self.args = args

        self.wavenet = WaveNet(args.layer_size, args.stack_size,
                               args.in_channels, args.res_channels,
                               lr=args.lr, decay=args.lr_decay)

        train_dataset = MapsRandomDataset(
            args.train_data_dir, self.wavenet.receptive_fields, args.sample_size, args.in_channels
        )
        val_dataset = MapsRandomDataset(
            args.val_data_dir, self.wavenet.receptive_fields, args.sample_size, args.in_channels
        )

        self.train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True,
                                                        num_workers=args.num_workers)
        self.val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=args.batch_size, shuffle=True,
                                                      num_workers=args.num_workers)

        self.init_step = 0
        if self.args.cont:
            self.init_step = self.wavenet.load(args.model_dir, -1) + 1
        self._writer = SummaryWriter(args.log_dir)

    def run(self):
        for epoch_idx in range(self.init_step, self.args.epochs):
            print(f"Epoch {epoch_idx}")
            train_loss = self._epoch(epoch_idx, "train")
            # save model after each epoch
            val_loss = self._epoch(epoch_idx, "val")
            if self.wavenet.scheduler:
                self.wavenet.scheduler.step()
                self._writer.add_scalar(
                    "Learning rate",
                    self.wavenet.scheduler.get_last_lr()[0],
                    epoch_idx
                )
            print(f"Epoch: {epoch_idx}. Train loss: {train_loss}, val loss: {val_loss}")
            self.wavenet.save(self.args.model_dir, step=epoch_idx)

    def _epoch(self, epoch_idx, mode="train"):  # train or val
        _set = self.train_loader if mode == "train" else self.val_loader
        _log_cat = "Loss/train" if mode == "train" else "Loss/val"
        losses = []
        for inputs, targets in tqdm(_set):
            inputs = inputs.cuda()
            targets = targets.cuda()
            if np.any(np.isnan(inputs.cpu().numpy())) or np.any(np.isnan(targets.cpu().numpy())):
                continue
            if mode == "train":
                loss = self.wavenet.train(inputs, targets)
            else:  # evaluate
                loss = self.wavenet.eval(inputs, targets)
            losses.append(loss.cpu())
        _loss = np.mean(np.array(losses))
        self._writer.add_scalar(_log_cat, _loss, epoch_idx)
        return _loss


def prepare_output_dir(args):
    args.log_dir = Path(args.output_dir, "log")
    args.model_dir = Path(args.output_dir, "model")
    args.test_output_dir = Path(args.output_dir, "test")

    args.log_dir.mkdir(exist_ok=True, parents=True)
    args.model_dir.mkdir(exist_ok=True, parents=True)
    args.test_output_dir.mkdir(exist_ok=True, parents=True)


def save_args(args):
    out_path = Path(args.output_dir, "config.pickle")
    with open(out_path, "wb") as file:
        pickle.dump(args, file, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    _args = config.parse_args()

    prepare_output_dir(_args)
    save_args(_args)

    trainer = Trainer(_args)
    trainer.run()
