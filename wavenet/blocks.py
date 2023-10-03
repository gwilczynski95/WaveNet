import numpy as np
import torch
import torch.nn.functional as F

from wavenet.exceptions import InputSizeError


class CausalConv1d(torch.nn.Module):
    def __init__(self, channels_in, channels_out, dilation=1, **kwargs):
        super(CausalConv1d, self).__init__()

        self.kernel_size = 2
        self.pad = (self.kernel_size - 1) * dilation
        self.in_channels = channels_in
        self.out_channels = channels_out
        self.dilation = dilation

        self.conv = torch.nn.Conv1d(
            channels_in,
            channels_out,
            kernel_size=self.kernel_size,
            padding=0,
            dilation=dilation,
            **kwargs
        )

    def forward(self, x):
        x = F.pad(x, (self.pad, 0))
        return self.conv(x)


class ResidualBlock(torch.nn.Module):
    def __init__(self, res_channels, skip_channels, dilation):
        super(ResidualBlock, self).__init__()
        self.res_channels = res_channels
        self.dilated = CausalConv1d(
            res_channels, res_channels * 2, dilation=dilation
        )
        self.conv_res = torch.nn.Conv1d(
            res_channels, res_channels, 1
        )
        self.conv_skip = torch.nn.Conv1d(
            res_channels, skip_channels, 1
        )
        self.gate_tanh = torch.nn.Tanh()
        self.gate_sigmoid = torch.nn.Sigmoid()

    def forward(self, x, skip_size):
        out = self.dilated(x)

        gated_tanh = self.gate_tanh(out[:, :self.res_channels, :])
        gated_sigmoid = self.gate_sigmoid(out[:, self.res_channels:, :])
        gated = gated_sigmoid * gated_tanh

        output = self.conv_res(gated)
        output += x

        # skip connection
        skip = self.conv_skip(gated)
        skip = skip[:, :, -skip_size:]
        return output, skip


class ResidualStack(torch.nn.Module):
    def __init__(self, layer_size, stack_size, res_channels, skip_channels):
        """
        Stack residual blocks by layer and stack size
        :param layer_size: integer, 10 = layer[dilation=1, dilation=2, 4, 8, 16, 32, 64, 128, 256, 512]
        :param stack_size: integer, 5 = stack[layer1, layer2, layer3, layer4, layer5]
        :param res_channels: number of residual channel for input, output
        :param skip_channels: number of skip channel for output
        :return:
        """
        super(ResidualStack, self).__init__()

        self.layer_size = layer_size
        self.stack_size = stack_size

        self.res_blocks = self.stack_res_block(res_channels, skip_channels)

    @staticmethod
    def _residual_block(res_channels, skip_channels, dilation):
        block = ResidualBlock(res_channels, skip_channels, dilation)

        if torch.cuda.device_count() > 1:
            block = torch.nn.DataParallel(block)

        if torch.cuda.is_available():
            block.cuda()

        return block

    def build_dilations(self):
        dilations = []

        for s in range(0, self.stack_size):
            for l in range(0, self.layer_size):
                dilations.append(2 ** l)

        return dilations

    def stack_res_block(self, res_channels, skip_channels):
        """
        Prepare dilated convolution blocks by layer and stack size
        :return:
        """
        res_blocks = []
        dilations = self.build_dilations()

        for dilation in dilations:
            block = self._residual_block(res_channels, skip_channels, dilation)
            res_blocks.append(block)

        return torch.nn.ModuleList(res_blocks)

    def forward(self, x, skip_size):
        """
        :param x:
        :param skip_size: The last output size for loss and prediction
        :return:
        """
        output = x
        skip_connections = []

        for res_block in self.res_blocks:
            # output is the next input
            output, skip = res_block(output, skip_size)
            skip_connections.append(skip)

        return torch.stack(skip_connections)


class OutHead(torch.nn.Module):
    def __init__(self, channels):
        """
        The last network of WaveNet
        :param channels: number of channels for input and output
        :return:
        """
        super(OutHead, self).__init__()

        self.conv1 = torch.nn.Conv1d(channels, channels, 1)
        self.conv2 = torch.nn.Conv1d(channels, channels, 1)

        self.relu = torch.nn.ReLU()
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        output = self.relu(x)
        output = self.conv1(output)
        output = self.relu(output)
        output = self.conv2(output)

        return output


class WaveNetModel(torch.nn.Module):
    def __init__(self, layer_size, stack_size, in_channels, res_channels):
        """
        Stack residual blocks by layer and stack size
        :param layer_size: integer, 10 = layer[dilation=1, dilation=2, 4, 8, 16, 32, 64, 128, 256, 512]
        :param stack_size: integer, 5 = stack[layer1, layer2, layer3, layer4, layer5]
        :param in_channels: number of channels for input data. skip channel is same as input channel
        :param res_channels: number of residual channel for input, output
        :return:
        """
        super(WaveNetModel, self).__init__()

        self.receptive_fields = self.calc_receptive_fields(layer_size, stack_size)

        self.causal = CausalConv1d(in_channels, res_channels)

        self.res_stack = ResidualStack(layer_size, stack_size, res_channels, in_channels)

        self.out_head = OutHead(in_channels)

    @staticmethod
    def calc_receptive_fields(layer_size, stack_size):
        layers = [2 ** i for i in range(0, layer_size)] * stack_size
        num_receptive_fields = np.sum(layers)

        return int(num_receptive_fields)

    def calc_output_size(self, x):
        output_size = int(x.size(2)) - self.receptive_fields

        self.check_input_size(x, output_size)

        return output_size

    def check_input_size(self, x, output_size):
        if output_size < 1:
            raise InputSizeError(int(x.size(2)), self.receptive_fields, output_size)

    def forward(self, x):
        """
        The size of timestep(3rd dimention) has to be bigger than receptive fields
        :param x: Tensor[batch, timestep, channels]
        :return: Tensor[batch, timestep, channels]
        """
        output = x

        output_size = self.calc_output_size(output)

        output = self.causal(output)

        skip_connections = self.res_stack(output, output_size)

        output = torch.sum(skip_connections, dim=0)

        output = self.out_head(output)

        return output
