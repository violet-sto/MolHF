import torch
import torch.nn as nn
import torch.nn.functional as F
from models.rgcn import RGCN
from models.basic import ActNorm2D, InvRotationLU

def squeeze_adj(adj, stride):
    # the same as sum pool for hard assign_matrix
    n_node = adj.shape[-1]
    assign_matrix = torch.zeros(n_node, n_node//stride).to(adj.device)
    for i in range(n_node//stride):
        assign_matrix[i*stride:(i+1)*stride, i] = 1

    adj_new = torch.einsum(
        'ik,bekj-> beij', assign_matrix.transpose(0, 1), adj)
    adj_new = torch.einsum('beik,kj->beij', adj_new, assign_matrix)

    return adj_new


class GraphAffineCoupling(nn.Module):
    def __init__(self, num_layers, n_node, in_dim, hid_dim, mask_swap=False):
        super(GraphAffineCoupling, self).__init__()
        self.mask_swap = mask_swap

        self.net = RGCN(num_layers, n_node, in_dim, hid_dim, in_dim)

        self.net.linear2.weight.data.zero_()
        self.net.linear2.bias.data.zero_()

    def forward(self, input, adj):
        in_a, in_b = input.chunk(2, 2)

        if self.mask_swap:
            in_a, in_b = in_b, in_a

        log_s, t = self.net(in_a, adj).chunk(2, 2)
        log_s = F.silu(log_s)
        s = torch.sigmoid(log_s)
        out_b = (in_b + t) * s

        logdet = torch.sum(
            torch.log(torch.abs(s)).reshape(input.shape[0], -1), 1)

        if self.mask_swap:
            out = torch.cat([out_b, in_a], 2)
        else:
            out = torch.cat([in_a, out_b], 2)

        return out, logdet

    def reverse(self, output, adj):
        out_a, out_b = output.chunk(2, 2)
        if self.mask_swap:
            out_a, out_b = out_b, out_a

        log_s, t = self.net(out_a, adj).chunk(2, 2)
        log_s = F.silu(log_s)
        s = torch.sigmoid(log_s)
        in_b = out_b / s - t  # More stable, less error   s must not equal to 0!!!

        if self.mask_swap:
            result = torch.cat([in_b, out_a], 2)
        else:
            result = torch.cat([out_a, in_b], 2)

        return result

class Gaussianize(nn.Module):
    def __init__(self, num_layers, n_node, in_dim, hid_dim):
        super(Gaussianize, self).__init__()
        self.net = RGCN(num_layers, n_node, in_dim, hid_dim, in_dim)

        self.net.linear2.weight.data.zero_()
        self.net.linear2.bias.data.zero_()

    def forward(self, input, cond, adj):
        log_std, mean = self.net(cond, adj).chunk(2, 2)

        log_std = F.silu(log_std)
        std = torch.sigmoid(log_std)
        std = 1/std

        out = (input-mean)*std
        if std.sum() != std.sum():
            print("nan")
        logdet = torch.sum(torch.log(std).view(input.shape[0], -1), 1)

        return out, logdet

    def reverse(self, output, cond, adj):
        log_std, mean = self.net(cond, adj).chunk(2, 2)

        log_std = F.silu(log_std)
        std = torch.sigmoid(log_std)
        std = 1/std

        input = output/std+mean
        return input


class FlowOnGraph(nn.Module):
    def __init__(self, num_layers, n_node, in_dim, hid_dim, inv_rotate, mask_swap):
        super(FlowOnGraph, self).__init__()
        self.n_node = n_node
        self.in_dim = in_dim
        self.inv_rotate = inv_rotate
        self.actnorm = ActNorm2D(n_node)
        if inv_rotate:
            self.invrotate = InvRotationLU(in_dim)
            mask_swap = False
        self.coupling = GraphAffineCoupling(
            num_layers, n_node, in_dim, hid_dim, mask_swap=mask_swap)

    def forward(self, input, adj):  # (Batch,N,d) (Batch,b,N,N)
        out, logdet = self.actnorm(input)
        if self.inv_rotate:
            out, det = self.invrotate(out)
            logdet = logdet+det

        out, det = self.coupling(out, adj)
        logdet = logdet+det

        return out, logdet

    def reverse(self, output, adj):
        input = self.coupling.reverse(output, adj)
        if self.inv_rotate:
            input = self.invrotate.reverse(input)
        input = self.actnorm.reverse(input)
        return input


class BlockOnGraph(nn.Module):
    def __init__(self, stride, num_flows, num_layers, n_node, in_dim, hid_dim, inv_rotate, split, condition=True):
        super(BlockOnGraph, self).__init__()
        self.stride = stride
        self.n_node = n_node
        self.split = split
        self.flows = nn.ModuleList()
        for i in range(num_flows):
            self.flows.append(FlowOnGraph(
                num_layers, n_node//stride, in_dim*stride, hid_dim, inv_rotate, mask_swap=bool(i % 2)))
        self.condition = condition
        self.gaussianize = Gaussianize(
            num_layers, n_node//stride, in_dim*stride, hid_dim)

    def forward(self, input, adj):
        z_new = None
        out, adj_new = self._squeeze(input, adj)
        logdet = 0

        for flow in self.flows:
            out, det = flow(out, adj_new)
            logdet = logdet + det

        if self.split:
            out, z_new = out.chunk(2, 2)
            if self.condition:
                out, det = self.gaussianize(out, z_new, adj_new)
                logdet = logdet + det

        else:
            z_new = out

        return out, logdet, z_new, adj_new

    def reverse(self, output, adj, z):

        if self.split:
            if self.condition:
                output = self.gaussianize.reverse(output, z, adj)
            input = torch.cat((output, z), dim=2)
        else:
            input = z

        for flow in self.flows[::-1]:
            input = flow.reverse(input, adj)

        input = self._unsqueeze(input)

        return input

    def _squeeze(self, x, adj):
        batch_size, n_node, _ = x.shape
        fold = self.stride
        out = x.reshape(batch_size, n_node//fold, -1)
        adj_new = squeeze_adj(adj, fold)

        return out, adj_new

    def _unsqueeze(self, x):
        batch_size, n_node, _ = x.shape
        fold = self.stride
        out = x.reshape(batch_size, n_node*fold, -1)

        return out


class GraphFlow(nn.Module):
    def __init__(self, n_block, squeeze_fold, num_flow, num_layers, n_node, in_dim, hid_dim, inv_rotate, condition=True):
        super().__init__()
        self.n_block = n_block
        self.squeeze_fold = squeeze_fold
        self.num_layers = num_layers
        self.n_node = n_node
        self.in_dim = in_dim
        self.Blocks = nn.ModuleList()
        for i in range(self.n_block-1):
            if i == 0:
                fold = 1
            else:
                fold = squeeze_fold

            self.Blocks.append(BlockOnGraph(
                fold, num_flow, num_layers, n_node, in_dim, hid_dim, inv_rotate, split=True, condition=condition))
            n_node //= fold
            in_dim = in_dim * fold // 2

        self.Blocks.append(
            BlockOnGraph(squeeze_fold, num_flow, num_layers, n_node, in_dim, hid_dim, inv_rotate, split=False, condition=condition))

    def forward(self, input, adj):
        out = input
        logdet = 0
        z_outs = []
        for block in self.Blocks:
            out, det, z_new, adj = block(out, adj)
            z_outs.append(z_new)
            logdet = logdet + det

        assert sum([z[0].numel() for z in z_outs if z is not None]) == self.n_node * \
            self.in_dim, 'Dim of latent variables is not equal to dim of features!'

        return z_outs, logdet

    def reverse(self, output, adj):
        adjs = [adj]
        for i in range(self.n_block-1):
            adj = squeeze_adj(adj, self.squeeze_fold)
            adjs.append(adj)

        for idx, transform in enumerate(self.Blocks[::-1]):
            if idx == 0:
                input = transform.reverse(
                    None, adjs[-(idx+1)], output[-(idx+1)])

            else:
                input = transform.reverse(
                    input, adjs[-(idx+1)], output[-(idx+1)])

        return input
