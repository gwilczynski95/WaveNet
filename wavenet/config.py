import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--layer_size', type=int, default=10,
                    help='layer_size: 10 = layer[dilation=1, dilation=2, 4, 8, 16, 32, 64, 128, 256, 512]')
parser.add_argument('--stack_size', type=int, default=5,
                    help='stack_size: 5 = stack[layer1, layer2, layer3, layer4, layer5]')
parser.add_argument('--in_channels', type=int, default=256,
                    help='input channel size. mu-law encode factor, one-hot size')
parser.add_argument('--res_channels', type=int, default=512, help='number of channel for residual network')

parser.add_argument('--sample_rate', type=int, default=16000, help='Sampling rates for input sound')
parser.add_argument('--sample_size', type=int, default=100000, help='Sample size for training input')


def parse_args(is_training=True):
    if is_training:
        parser.add_argument('--train_data_dir', type=str, default='./test/data', help='Training data dir')
        parser.add_argument('--val_data_dir', type=str, default='./test/data', help='Validation data dir')
        parser.add_argument('--output_dir', type=str, default='./output', help='Output dir for saving model and etc')
        parser.add_argument('--epochs', type=int, default=100, help='Total number of epochs')
        parser.add_argument('--lr', type=float, default=0.0002, help='learning rate decay')
        parser.add_argument("--batch_size", type=int, default=2, help="Size of a batch")
        parser.add_argument("--lr_decay", type=float, default=0., help="Decay param, if 0 then training won't use lrs")
        parser.add_argument('--cont', action='store_true', help="If training should be continued")
        parser.add_argument("--num_workers", type=int, default=0, help="Num of workers for dataloader")
    else:
        parser.add_argument('--model_dir', type=str, required=True, help='Pre-trained model dir')
        parser.add_argument('--step', type=int, default=-1, help='A specific step of pre-trained model to use')
        parser.add_argument('--seed', type=str, required=True, help='A seed file to generate sound')
        parser.add_argument("--sec", type=int, default=7, help="How many seconds long audio")
        parser.add_argument("--temperature", type=float, default=0, help="Temperature for generating audio")

    return parser.parse_args()


def print_help():
    parser.print_help()
