import torch
import torch.nn as nn
import math


class DilatedConv1d(nn.Conv1d):
    """Dilated Conv1d."""
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1,
                 groups=1, bias=True, causal=True):
        """The init function of dilated Conv1D.

        Parameters
        ---------
        in_channels(int): number of channels of input (number of rows of input)
        out_channels(int): number of channels of output (number of kernels
        used)
        kernel_size(int): size of kernel
        dilation(int): dilation rate
        causal(bool): whether it is a causal convolution or not
        """
        self.causal = causal
        if causal:
            padding = (kernel_size - 1)*dilation
            # padding is on both sides!!!
        else:
            padding = (kernel_size - 1)*dilation//2
        super(DilatedConv1d, self).__init__(
            in_channels, out_channels,
            kernel_size, stride=1,
            padding=padding,
            dilation=dilation,
            groups=groups, bias=bias)

    def forward(self, inputs):
        output = super(DilatedConv1d, self).forward(inputs)
        if self.causal:
            # because padding on both side, removing trailing zero paddings
            output = output[:, :, :-self.padding[0]]
        return output


class LayerNorm(nn.Module):
    '''Layer normalization.'''
    def __init__(self, num_features, eps=1e-6):  # ,affine=True):
        """Layer Normalization.

        This class implement layer normalization, which is a
        sample wise activity normalization.

        Parameters
        ----------
        num_features (int): number of features (hidden units)
        eps (float): regularization term
        """
        super(LayerNorm, self).__init__()
        self.eps = eps
        self.num_features = num_features
        # self.affine = affine
        self.weight = nn.Parameter(torch.FloatTensor(num_features))
        # no. channels
        # floatTensor of size num_features
        self.bias = nn.Parameter(torch.FloatTensor(num_features))
        self.reset_parameters()

    def reset_parameters(self):
        # adapted std
        self.weight.data.fill_(1.)
        # adapted mean
        self.bias.data.fill_(0.)

    def forward(self, inputs):
        # k = batch size, n = channels, m=feature length
        k, n, _ = list(inputs.size())

        # calculate the mean, on 2nd axis
        mean = inputs.mean(2, keepdim=True).expand_as(inputs)

        # subtract the mean
        input_centered = inputs - mean

        # calculate standard deviation
        std = input_centered.pow(2).mean(2, keepdim=True).add(self.eps).sqrt()

        # divide the std
        output = input_centered / std.expand_as(inputs)

        # reshape weights to match with the input
        w = self.weight.view(1, -1, 1).expand_as(output)
        b = self.bias.view(1, -1, 1).expand_as(output)

        return output*w+b


class ResBlock(nn.Module):
    '''Redidual block.
    '''
    def __init__(self, in_channels, d_channels=None, out_channels=None,
                 kernel_size=3, dilation=1, causal=True, mode="encoder"):
        """The ResNet block.

        Parameters
        ----------
        in_channels (int) = number of input channels
        d_channels (int) = number of intermediate channels
        out_channels (int) = number of output channels
        kernel_size (int) = size of kernel_size
        dilation (int) = dilation rate
        causal(bool) = causal convolution or not
        """
        super(ResBlock, self).__init__()
        out_channels = out_channels or in_channels
        d_channels = d_channels or in_channels // 2
        self.mode = mode

        # layer normalization layer
        self.layernorm1 = LayerNorm(num_features=in_channels)
        self.layernorm2 = LayerNorm(num_features=d_channels)
        self.layernorm3 = LayerNorm(num_features=d_channels)

        # input 1x1 convolution
        self.conv_in = nn.Conv1d(in_channels=in_channels,
                                 out_channels=d_channels,
                                 kernel_size=1)

        # dilation convolution in the middle
        self.conv_mid = DilatedConv1d(in_channels=d_channels,
                                      out_channels=d_channels,
                                      kernel_size=kernel_size,
                                      dilation=dilation,
                                      causal=causal)

        # output 1x1 convolution
        self.conv_out = nn.Conv1d(in_channels=d_channels,
                                  out_channels=out_channels,
                                  kernel_size=1)

        # activation
        self.relu = nn.ReLU()

    def forward(self, inputs):
        out = self.layernorm1(inputs.contiguous())
        out = self.conv_in(self.relu(out))
        out = self.layernorm2(out)
        out = self.conv_mid(self.relu(out))
        out = self.layernorm3(out)
        out = self.conv_out(self.relu(out))
        if self.mode == "encoder":
            out = (out+inputs) * math.sqrt(0.5)
        return out


class EncoderDecoder(nn.Module):
    """Encoder / Decoder, nn.Module approach."""
    def __init__(self, num_channels, num_sets=6,
                 dilation_rates=[1, 2, 4, 8, 16], kernel_size=3,
                 block_type='decoder'):
        """ByteNet Encoder / Decoder.

        Parameters
        ----------
        num_channels (int) = number of channels
        num_sets (int) = number of ByteNet architectures
        dilation_rates (list) = list of dilation rates
        kernel_size (int) = the size of the kernel
        block_type (str) = "encoder" (not causal) or "decoder" (causal)
        """
        super(EncoderDecoder, self).__init__()
        causal = False if type == "encoder" else True

        self.layers = []
        for s_idx in range(num_sets):
            for d_rate in dilation_rates:
                self.layers.append(ResBlock(num_channels,
                                            kernel_size=kernel_size,
                                            dilation=d_rate,
                                            causal=causal))
        self.layers = nn.Sequential(*self.layers)

    def forward(self, inputs):
        """carry out encoder / decoder."""
        out = inputs
        for layer in self.layers:
            out = layer(out)

        return out
