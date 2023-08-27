import torch
import torch.nn as nn
import torch.nn.functional as F
from models.basic import ZeroConv2d, ActNorm, InvConv2dLU, InvConv2d
from models.cc import CrissCrossAttention

class AffineCoupling(nn.Module):
    def __init__(self, in_channel, filter_size, affine=True, mask_swap=False):
        super().__init__()

        self.affine = affine
        self.mask_swap = mask_swap

        self.net = nn.Sequential(
            nn.Conv2d(in_channel // 2, filter_size, 3, padding=1),
            nn.BatchNorm2d(filter_size),
            nn.ReLU(inplace=True),
            CrissCrossAttention(filter_size, filter_size),
            nn.BatchNorm2d(filter_size),
            nn.ReLU(inplace=True),
            nn.Conv2d(filter_size, in_channel, 3, padding=1),
        )
        self.net[0].weight.data.normal_(0, 0.05)
        self.net[0].bias.data.zero_()

    def forward(self, input):
        in_a, in_b = input.chunk(2, 1)

        if self.mask_swap:
            in_a, in_b = in_b, in_a

        if self.affine:
            log_s, t = self.net(in_a).chunk(2, 1)
            log_s = F.silu(log_s)
            s = torch.sigmoid(log_s)
            out_b = (in_b + t) * s
            if s.sum() != s.sum():
                print("nan")
            logdet = torch.sum(torch.log(s).view(input.shape[0], -1), 1)

        else:
            net_out = self.net(in_a)
            out_b = in_b + net_out
            logdet = None

        if self.mask_swap:
            result = torch.cat([out_b, in_a], 1)
        else:
            result = torch.cat([in_a, out_b], 1)

        return result, logdet

    def reverse(self, output):
        out_a, out_b = output.chunk(2, 1)
        if self.mask_swap:
            out_a, out_b = out_b, out_a

        if self.affine:
            log_s, t = self.net(out_a).chunk(2, 1)
            log_s = F.silu(log_s)
            s = torch.sigmoid(log_s)
            if s.sum() != s.sum():
                print("nan")
            in_b = out_b / s - t

        else:
            net_out = self.net(out_a)
            in_b = out_b - net_out

        if self.mask_swap:
            result = torch.cat([in_b, out_a], 1)
        else:
            result = torch.cat([out_a, in_b], 1)

        return result


class Flow(nn.Module):
    def __init__(self, in_channel, filter_size, affine=True, inv_conv=True, conv_lu=True, mask_swap=False):
        super().__init__()

        self.inv_conv = inv_conv
        self.actnorm = ActNorm(in_channel)

        if inv_conv:
            mask_swap = False
            if conv_lu:
                self.invconv = InvConv2dLU(in_channel)

            else:
                self.invconv = InvConv2d(in_channel)

        self.coupling = AffineCoupling(
            in_channel, filter_size=filter_size, affine=affine, mask_swap=mask_swap)

    def forward(self, input):
        out, logdet = self.actnorm(input)
        if self.inv_conv:
            out, det1 = self.invconv(out)
            logdet = logdet + det1
        out, det2 = self.coupling(out)

        if det2 is not None:
            logdet = logdet + det2
        return out, logdet

    def reverse(self, output):
        input = self.coupling.reverse(output)
        if self.inv_conv:
            input = self.invconv.reverse(input)
        input = self.actnorm.reverse(input)

        return input


class Gaussianize(nn.Module):
    def __init__(self, in_channel, filter_size):
        super(Gaussianize, self).__init__()
        self.net = ZeroConv2d(in_channel // 2, in_channel)

    def forward(self, input, cond):
        log_std, mean = self.net(cond).chunk(2, 1)
        std = torch.exp(log_std)
        std = 1/std

        out = (input-mean)*std
        if std.sum() != std.sum():
            print("nan")
        logdet = torch.sum(torch.log(std).view(input.shape[0], -1), 1)

        return out, logdet

    def reverse(self, output, cond):
        log_std, mean = self.net(cond).chunk(2, 1)

        std = torch.exp(log_std)
        std = 1/std

        input = output/std+mean

        return input


class Block(nn.Module):
    def __init__(self, in_channel, filter_size, n_flow, squeeze_fold=2, split=True, affine=True, inv_conv=True, conv_lu=True, condition=True):
        super().__init__()
        self.squeeze_fold = squeeze_fold
        self.squeeze_dim = in_channel * squeeze_fold**2

        self.flows = nn.ModuleList()
        for i in range(n_flow):
            self.flows.append(
                Flow(self.squeeze_dim, filter_size, affine=affine, inv_conv=inv_conv, conv_lu=conv_lu, mask_swap=i % 2))

        self.split = split
        self.condition = condition
        self.gaussianize = Gaussianize(self.squeeze_dim, filter_size)

    def forward(self, input):
        out = self._squeeze(input)
        logdet = 0

        for flow in self.flows:
            out, det = flow(out)
            logdet = logdet + det

        if self.split:
            out, z_new = out.chunk(2, 1)
            if self.condition:
                out, det = self.gaussianize(out, z_new)
                logdet = logdet + det

        else:
            z_new = out
        return out, logdet, z_new

    def reverse(self, output, z):
        if self.split:
            if self.condition:
                output = self.gaussianize.reverse(output, z)
            input = torch.cat([output, z], 1)

        else:
            input = z

        for flow in self.flows[::-1]:
            input = flow.reverse(input)

        unsqueezed = self._unsqueeze(input)
        return unsqueezed

    def _squeeze(self, x):
        """Trade spatial extent for channels. In forward direction, convert each
        1x4x4 volume of input into a 4x1x1 volume of output.

        Args:
            x (torch.Tensor): Input to squeeze or unsqueeze.
            reverse (bool): reverse the operation, i.e., unsqueeze.

        Returns:
            x (torch.Tensor): Squeezed or unsqueezed tensor.
        """
        # b, c, h, w = x.size()
        assert len(x.shape) == 4
        b_size, n_channel, height, width = x.shape
        fold = self.squeeze_fold

        squeezed = x.view(b_size, n_channel, height // fold,
                          fold,  width // fold,  fold)
        squeezed = squeezed.permute(0, 1, 3, 5, 2, 4).contiguous()
        out = squeezed.view(b_size, n_channel * fold * fold,
                            height // fold, width // fold)
        return out

    def _unsqueeze(self, x):
        assert len(x.shape) == 4
        b_size, n_channel, height, width = x.shape
        fold = self.squeeze_fold
        unsqueezed = x.view(b_size, n_channel //
                            (fold * fold), fold, fold, height, width)
        unsqueezed = unsqueezed.permute(0, 1, 4, 2, 5, 3).contiguous()
        out = unsqueezed.view(b_size, n_channel //
                              (fold * fold), height * fold, width * fold)
        return out


class CC_Glow(nn.Module):
    def __init__(
        self, n_block, in_channel, filter_size, n_flow, squeeze_fold, affine=True, inv_conv=True, conv_lu=True, condition=True):
        super().__init__()
        self.n_block = n_block-1
        self.squeeze_fold = squeeze_fold
        self.blocks = nn.ModuleList()
        n_channel = in_channel
        for i in range(self.n_block-1):
            self.blocks.append(
                Block(n_channel, filter_size, n_flow, squeeze_fold, split=True, affine=affine, inv_conv=inv_conv, conv_lu=conv_lu, condition=condition))
            n_channel = n_channel*squeeze_fold**2//2

        self.blocks.append(
            Block(n_channel, filter_size, n_flow, squeeze_fold, split=False, affine=affine, inv_conv=inv_conv, conv_lu=conv_lu, condition=condition))

    def forward(self, input):
        logdet = 0
        out = input
        z_outs = []

        for block in self.blocks:
            out, det, z_new = block(out)
            z_outs.append(z_new)
            logdet = logdet + det

        return z_outs, logdet

    def reverse(self, z_list):
        for i, block in enumerate(self.blocks[::-1]):
            if i == 0:
                input = block.reverse(
                    z_list[-1], z_list[-1])

            else:
                input = block.reverse(
                    input, z_list[-(i + 1)])

        return input
