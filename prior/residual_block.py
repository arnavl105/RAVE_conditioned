import torch
import torch.nn as nn
import cached_conv as cc


class ResidualBlock(nn.Module):

    def __init__(self, res_size, skp_size, kernel_size, dilation):
        super().__init__()
        fks = (kernel_size - 1) * dilation + 1

        self.dconv = cc.Conv1d(
            res_size,
            2 * res_size,
            kernel_size,
            padding=(fks - 1, 0),
            dilation=dilation,
        )

        self.rconv = nn.Conv1d(res_size, res_size, 1)
        self.sconv = nn.Conv1d(res_size, skp_size, 1)

        # Add a 1x1 convolution layer for the onset strength envelope
        self.cconv = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=1, stride=2048//res_size)

    def forward(self, x, skp, onset_strength):
        res = x.clone()

        x = self.dconv(x)

        xa, xb = torch.split(x, x.shape[1] // 2, 1)

        if onset_strength is not None:
            onset_conv = self.cconv(onset_strength)
            xa = xa + onset_conv.t()
            xb = xb + onset_conv.t()

        x = torch.sigmoid(xa) * torch.tanh(xb)
        r_conv = self.rconv(x)
        res = res + r_conv
        skp = skp + self.sconv(x)
        return res, skp
