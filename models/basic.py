from scipy import linalg as la
import numpy as np
from torch.nn import functional as F
from torch import nn
import torch
def logabs(x): return torch.log(torch.abs(x))

# Basic invertible layers
class ActNorm(nn.Module):
    def __init__(self, in_channel, logdet=True):
        super().__init__()

        self.loc = nn.Parameter(torch.zeros(1, in_channel, 1, 1))
        self.scale = nn.Parameter(torch.ones(1, in_channel, 1, 1))

        self.register_buffer('initialized', torch.tensor(0, dtype=torch.uint8))
        self.logdet = logdet

    def initialize(self, input):
        with torch.no_grad():
            flatten = input.permute(
                1, 0, 2, 3).contiguous().view(input.shape[1], -1)
            mean = (
                flatten.mean(1)
                .unsqueeze(1)
                .unsqueeze(2)
                .unsqueeze(3)
                .permute(1, 0, 2, 3)
            )
            std = (
                flatten.std(1)
                .unsqueeze(1)
                .unsqueeze(2)
                .unsqueeze(3)
                .permute(1, 0, 2, 3)
            )

            self.loc.data.copy_(-mean)
            self.scale.data.copy_(1 / (std + 1e-6))

    def forward(self, input):
        _, _, height, width = input.shape

        if self.initialized.item() == 0:
            self.initialize(input)
            self.initialized.fill_(1)

        out = self.scale * (input + self.loc)
        logdet = height * width * torch.sum(logabs(self.scale))

        if self.logdet:
            return out, logdet

        else:
            return out

    def reverse(self, output):
        return output / self.scale - self.loc


class ActNormSym(nn.Module):
    def __init__(self, in_channel, logdet=True):
        super().__init__()

        self.loc = nn.Parameter(torch.zeros(1, in_channel, 1, 1))
        self.scale = nn.Parameter(torch.ones(1, in_channel, 1, 1))

        self.register_buffer('initialized', torch.tensor(0, dtype=torch.uint8))
        self.logdet = logdet

    def initialize(self, input):
        with torch.no_grad():
            n_node = input.shape[-1]
            tri_index = torch.tril_indices(n_node, n_node, -1)
            flatten = input[:, :, tri_index[0], tri_index[1]].permute(
                1, 0, 2).contiguous().view(input.shape[1], -1)

            mean = (
                flatten.mean(1)
                .unsqueeze(1)
                .unsqueeze(2)
                .unsqueeze(3)
                .permute(1, 0, 2, 3)
            )
            std = (
                flatten.std(1)
                .unsqueeze(1)
                .unsqueeze(2)
                .unsqueeze(3)
                .permute(1, 0, 2, 3)
            )

            self.loc.data.copy_(-mean)
            self.scale.data.copy_(1 / (std + 1e-6))

    def forward(self, input):
        _, _, height, width = input.shape

        if self.initialized.item() == 0:
            self.initialize(input)
            self.initialized.fill_(1)

        out = self.scale * (input + self.loc)
        logdet = height*(width-1)//2 * torch.sum(logabs(self.scale))

        if self.logdet:
            return out, logdet

        else:
            return out

    def reverse(self, output):
        return output / self.scale - self.loc


# class My_ActNorm(nn.Module):
#     #! Not work
#     def __init__(self, in_channel, logdet=True):
#         super().__init__()

#         self.loc = nn.Parameter(torch.zeros(1, 1, in_channel, in_channel))
#         self.scale = nn.Parameter(torch.ones(1, 1, in_channel, in_channel))

#         self.register_buffer('initialized', torch.tensor(0, dtype=torch.uint8))
#         self.logdet = logdet

#     def initialize(self, input):
#         with torch.no_grad():
#             mean = input.mean((0, 1)).unsqueeze(0).unsqueeze(1)
#             std = input.std((0, 1)).unsqueeze(0).unsqueeze(1)

#             self.loc.data.copy_(-mean)
#             self.scale.data.copy_(1 / (std + 1e-6))

#     def forward(self, input):
#         _, channel, _, _ = input.shape

#         if self.initialized.item() == 0:
#             self.initialize(input)
#             self.initialized.fill_(1)

#         out = self.scale * (input + self.loc)
#         logdet = channel * torch.sum(logabs(self.scale))

#         if self.logdet:
#             return out, logdet

#         else:
#             return out

#     def reverse(self, output):
#         return output / self.scale - self.loc


class ActNorm2D(nn.Module):
    def __init__(self, in_dim, logdet=True):
        super().__init__()

        self.loc = nn.Parameter(torch.zeros(1, in_dim, 1))
        self.scale = nn.Parameter(torch.ones(1, in_dim, 1))

        self.register_buffer('initialized', torch.tensor(0, dtype=torch.uint8))
        self.logdet = logdet

    def initialize(self, input):
        with torch.no_grad():
            flatten = input.permute(
                1, 0, 2).contiguous().view(input.shape[1], -1)
            mean = (
                flatten.mean(1)
                .unsqueeze(1)
                .unsqueeze(2)
                .permute(1, 0, 2)
            )
            std = (
                flatten.std(1)
                .unsqueeze(1)
                .unsqueeze(2)
                .permute(1, 0, 2)
            )

            self.loc.data.copy_(-mean)
            self.scale.data.copy_(1 / (std + 1e-6))

    def forward(self, input):
        _, _, height = input.shape

        if self.initialized.item() == 0:
            self.initialize(input)
            self.initialized.fill_(1)

        log_abs = logabs(self.scale)

        logdet = height * torch.sum(log_abs)

        if self.logdet:
            return self.scale * (input + self.loc), logdet

        else:
            return self.scale * (input + self.loc)

    def reverse(self, output):
        return output / self.scale - self.loc


class My_ActNorm2D(nn.Module):
    def __init__(self, in_dim, logdet=True):
        super().__init__()

        self.loc = nn.Parameter(torch.zeros(1, 1, in_dim))
        self.scale = nn.Parameter(torch.ones(1, 1, in_dim))

        self.register_buffer('initialized', torch.tensor(0, dtype=torch.uint8))
        self.logdet = logdet

    def initialize(self, input):
        with torch.no_grad():
            flatten = input.permute(
                2, 0, 1).contiguous().view(input.shape[2], -1)
            mean = (
                flatten.mean(1)
                .unsqueeze(1)
                .unsqueeze(2)
                .permute(1, 2, 0)
            )
            std = (
                flatten.std(1)
                .unsqueeze(1)
                .unsqueeze(2)
                .permute(1, 2, 0)
            )

            self.loc.data.copy_(-mean)
            self.scale.data.copy_(1 / (std + 1e-6))

    def forward(self, input):
        _, width, _ = input.shape

        if self.initialized.item() == 0:
            self.initialize(input)
            self.initialized.fill_(1)

        log_abs = logabs(self.scale)

        logdet = width * torch.sum(log_abs)

        if self.logdet:
            return self.scale * (input + self.loc), logdet

        else:
            return self.scale * (input + self.loc)

    def reverse(self, output):
        return output / self.scale - self.loc


class InvConv2d(nn.Module):
    """
    Invertible 1*1 conv
    """

    def __init__(self, in_channel):
        super().__init__()

        weight = torch.randn(in_channel, in_channel)
        q, _ = torch.linalg.qr(weight)  # generate random rotation matrix
        weight = q.unsqueeze(2).unsqueeze(3)
        self.weight = nn.Parameter(weight)

    def forward(self, input):
        _, _, height, width = input.shape

        out = F.conv2d(input, self.weight)
        logdet = (
            height * width *
            torch.linalg.slogdet(self.weight.squeeze().double())[1].float()
        )

        return out, logdet

    def reverse(self, output):
        return F.conv2d(
            output, self.weight.squeeze().inverse().unsqueeze(2).unsqueeze(3)
        )


class InvConv2dLU(nn.Module):
    """
    Invertible 1*1 conv with LU decomposition
    """

    def __init__(self, in_channel, symmetry=False):
        super().__init__()

        weight = np.random.randn(in_channel, in_channel)
        q, _ = la.qr(weight)
        w_p, w_l, w_u = la.lu(q.astype(np.float32))
        w_s = np.diag(w_u)
        w_u = np.triu(w_u, 1)
        u_mask = np.triu(np.ones_like(w_u), 1)
        l_mask = u_mask.T

        w_p = torch.from_numpy(w_p)
        w_l = torch.from_numpy(w_l)
        w_s = torch.from_numpy(w_s)
        w_u = torch.from_numpy(w_u)

        self.symmetry = symmetry
        self.register_buffer('w_p', w_p)
        self.register_buffer('u_mask', torch.from_numpy(u_mask))
        self.register_buffer('l_mask', torch.from_numpy(l_mask))
        self.register_buffer('s_sign', torch.sign(w_s))
        self.register_buffer('l_eye', torch.eye(l_mask.shape[0]))
        self.w_l = nn.Parameter(w_l)
        self.w_s = nn.Parameter(logabs(w_s))
        self.w_u = nn.Parameter(w_u)

    def forward(self, input):
        _, _, height, width = input.shape

        weight = self.calc_weight()

        out = F.conv2d(input, weight)

        if self.symmetry:
            logdet = height * (width-1)/2 + torch.sum(self.w_s)
        else:
            logdet = height * width * torch.sum(self.w_s)

        return out, logdet

    def calc_weight(self):
        weight = (
            self.w_p
            @ (self.w_l * self.l_mask + self.l_eye)
            @ ((self.w_u * self.u_mask) + torch.diag(self.s_sign * torch.exp(self.w_s)))
        )  # @ matrix multiplication

        return weight.unsqueeze(2).unsqueeze(3)

    def reverse(self, output):
        weight = self.calc_weight()

        return F.conv2d(output, weight.squeeze().inverse().unsqueeze(2).unsqueeze(3))


class InvConv2dLUSym(nn.Module):
    """
    Invertible 1*1 conv with LU decomposition
    """

    def __init__(self, in_channel):
        super().__init__()

        weight = np.random.randn(in_channel, in_channel)
        q, _ = la.qr(weight)
        w_p, w_l, w_u = la.lu(q.astype(np.float32))
        w_s = np.diag(w_u)
        w_u = np.triu(w_u, 1)
        u_mask = np.triu(np.ones_like(w_u), 1)
        l_mask = u_mask.T

        w_p = torch.from_numpy(w_p)
        w_l = torch.from_numpy(w_l)
        w_s = torch.from_numpy(w_s)
        w_u = torch.from_numpy(w_u)

        self.register_buffer('w_p', w_p)
        self.register_buffer('u_mask', torch.from_numpy(u_mask))
        self.register_buffer('l_mask', torch.from_numpy(l_mask))
        self.register_buffer('s_sign', torch.sign(w_s))
        self.register_buffer('l_eye', torch.eye(l_mask.shape[0]))
        self.w_l = nn.Parameter(w_l)
        self.w_s = nn.Parameter(logabs(w_s))
        self.w_u = nn.Parameter(w_u)

    def forward(self, input):
        _, _, height, width = input.shape

        weight = self.calc_weight()

        out = F.conv2d(input, weight)
        logdet = height*(width-1)/2 * torch.sum(self.w_s)

        return out, logdet

    def calc_weight(self):
        weight = (
            self.w_p
            @ (self.w_l * self.l_mask + self.l_eye)
            @ ((self.w_u * self.u_mask) + torch.diag(self.s_sign * torch.exp(self.w_s)))
        )  # @ matrix multiplication

        return weight.unsqueeze(2).unsqueeze(3)

    def reverse(self, output):
        weight = self.calc_weight()

        return F.conv2d(output, weight.squeeze().inverse().unsqueeze(2).unsqueeze(3))


class TensorLinear(nn.Module):
    def __init__(self, in_size, out_size, bias=True):
        super(TensorLinear, self).__init__()
        self.in_size = in_size
        self.out_size = out_size
        # Warning: differential initialization from Chainer
        self.linear = nn.Linear(in_size, out_size, bias)
        self.linear.weight.data.normal_(0, 0.05)
        self.linear.bias.data.zero_()

    def forward(self, x):
        if len(x.shape) == 4:
            h = x.permute(0, 2, 3, 1)
            h = self.linear(h)
            h = h.permute(0, 3, 1, 2)
        elif len(x.shape) == 3:
            h = x.permute(0, 2, 1)
            h = self.linear(h)
            h = h.permute(0, 2, 1)

        return h


class InvRotationLU(nn.Module):
    def __init__(self, dim):
        super(InvRotationLU, self).__init__()
        # (9*9)  * (bs*9*5)
        weight = np.random.randn(dim, dim)  # (9,9)
        q, _ = la.qr(weight)
        w_p, w_l, w_u = la.lu(q.astype(np.float32))
        w_s = np.diag(w_u)
        w_u = np.triu(w_u, 1)
        u_mask = np.triu(np.ones_like(w_u), 1)
        l_mask = u_mask.T

        w_p = torch.from_numpy(w_p)  # a Permutation matrix
        w_l = torch.from_numpy(w_l)  # L matrix from PLU
        w_s = torch.from_numpy(w_s)  # diagnal of the U matrix from PLU
        w_u = torch.from_numpy(w_u)  # u - dianal of the U matrix from PLU

        self.register_buffer('w_p', w_p)  # (12,12)
        self.register_buffer('u_mask', torch.from_numpy(
            u_mask))  # (12,12) upper 1 with 0 diagnal
        self.register_buffer('l_mask', torch.from_numpy(
            l_mask))  # (12,12) lower 1 with 0 diagnal
        # (12,)  # sign of the diagnal of the U matrix from PLU
        self.register_buffer('s_sign', torch.sign(w_s))
        self.register_buffer('l_eye', torch.eye(
            l_mask.shape[0]))  # (12,12) 1 diagnal
        self.w_l = nn.Parameter(w_l)  # (12,12)
        self.w_s = nn.Parameter(logabs(w_s))  # (12, )
        self.w_u = nn.Parameter(w_u)  # (12,12)

    def forward(self, input):
        bs, height, width = input.shape  # (bs, 9, 5)

        weight = self.calc_weight()  # 9,9

        # out = F.conv2d(input, weight)  # (2,12,32,32), (12,12,1,1) --> (2,12,32,32)
        # logdet = height * width * torch.sum(self.w_s)

        # (1, 9,9) * (bs, 9, 5) --> (bs, 9, 5)
        # out = torch.matmul(weight, input)
        out = torch.matmul(input, weight)

        logdet = height * torch.sum(self.w_s)

        return out, logdet

    def calc_weight(self):
        weight = (
            self.w_p
            @ (self.w_l * self.l_mask + self.l_eye)
            @ ((self.w_u * self.u_mask) + torch.diag(self.s_sign * torch.exp(self.w_s)))
        )
        # weight = torch.matmul(torch.matmul(self.w_p, (self.w_l * self.l_mask + self.l_eye)),
        #              ((self.w_u * self.u_mask) + torch.diag(self.s_sign * torch.exp(self.w_s))))
        # weight = self.w_p
        return weight.unsqueeze(0)  # weight.unsqueeze(2).unsqueeze(3)

    def reverse(self, output):
        weight = self.calc_weight()
        # return weight.inverse() @ output
        return torch.matmul(output, weight.inverse())
        # return F.conv2d(output, weight.squeeze().inverse().unsqueeze(2).unsqueeze(3))  #np.linalg.det(weight.data.numpy())


class InvRotation(nn.Module):
    def __init__(self, dim):
        super().__init__()

        weight = torch.randn(dim, dim)
        q, _ = torch.linalg.qr(weight)
        weight = q.unsqueeze(0)
        self.weight = nn.Parameter(weight)

    def forward(self, input):
        _, height, width = input.shape

        # out = F.conv2d(input, self.weight)
        out = self.weight @ input
        logdet = (
            width *
            torch.linalg.slogdet(self.weight.squeeze().double())[1].float()
        )

        return out, logdet

    def reverse(self, output):
        return self.weight.squeeze().inverse().unsqueeze(0) @ output


# Basic non-invertible layers in coupling _s_t_function, or for transforming Gaussian distribution
class ZeroConv2d(nn.Module):
    def __init__(self, in_channel, out_channel, padding=1):
        super().__init__()

        self.conv = nn.Conv2d(in_channel, out_channel, 3,
                              padding=0)  # in:512, out:12
        self.conv.weight.data.zero_()  # (12,512,3,3)
        self.conv.bias.data.zero_()  # 12
        self.scale = nn.Parameter(torch.zeros(
            1, out_channel, 1, 1))  # (1,12,1,1)

    def forward(self, input):
        # input: (2,512,32,32) --> (2,512,34,34)
        out = F.pad(input, [1, 1, 1, 1], value=1)
        out = self.conv(out)  # (2,12,32,32)
        # (2,12,32,32) * (1,12,1,1) = (2,12,32,32)
        out = out * torch.exp(self.scale * 3)

        return out

class ZeroMLP(nn.Module):
    def __init__(self, in_channel, hid_channel, out_channel):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(in_channel, hid_channel),
            nn.ReLU(),
            nn.Linear(hid_channel, out_channel)
        )
        self.net[0].weight.data.zero_() 
        self.net[0].bias.data.zero_()  
        self.net[2].weight.data.zero_() 
        self.net[2].bias.data.zero_() 
        self.scale = nn.Parameter(torch.zeros(
            1, 1, out_channel))  

    def forward(self, input, adj):
        out = self.net(input) 
        out = out * torch.exp(self.scale * 3)

        return out


class Rescale(nn.Module):
    def __init__(self):
        super(Rescale, self).__init__()
        self.weight = nn.Parameter(torch.zeros([1]))

    def forward(self, x):
        if torch.isnan(torch.exp(self.weight)).any():
            print(self.weight)
            raise RuntimeError('Rescale factor has NaN entries')

        x = torch.exp(self.weight) * x
        return x


def test_ZeroConv2d():
    in_channel = 1
    out_channel = 2

    x = torch.ones(2, 1, 5, 5)
    net = ZeroConv2d(in_channel, out_channel)
    y = net(x)
    print('x.shape:', x.shape)
    print(x)
    print('y.shape', y.shape)
    print(y)


def test_actnorm():
    in_channel = 4

    x = torch.ones(32, 4, 38, 38)
    net = ActNorm(in_channel)
    y = net(x)
    print('x.shape:', x.shape)
    print(x)
    print('y.shape', y[0].shape)
    print(y[0])


if __name__ == '__main__':
    torch.manual_seed(0)
    test_actnorm()
