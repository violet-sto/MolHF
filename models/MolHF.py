import torch
import torch.nn as nn
import math
from models.cc_glow import CC_Glow
from models.graphflow import GraphFlow
from copy import deepcopy

class MolHF(nn.Module):
    def __init__(self, data_config, args):
        super(MolHF, self).__init__()
        self.args = args
        self.device = args.device
        self.max_size = data_config['max_size']
        self.node_dim = data_config['node_dim']
        self.bond_dim = data_config['bond_dim']
        self.latent_node_length = self.max_size*self.node_dim
        self.latent_edge_length = self.max_size**2*self.bond_dim

        if args.learn_prior:
            self.prior_ln_var = nn.Parameter(torch.zeros(1))
        else:
            self.register_buffer('prior_ln_var', torch.zeros(1))

        self.bond_model = CC_Glow(
            n_block=args.n_block,
            in_channel=self.bond_dim,
            filter_size=args.filter_size,
            n_flow=args.b_num_flows,
            squeeze_fold=args.squeeze_fold,
            inv_conv=args.inv_conv,
            condition=args.condition
        )

        self.atom_model = GraphFlow(
            n_block=args.n_block,
            squeeze_fold=args.squeeze_fold,
            num_flow=args.a_num_flows,
            num_layers=args.num_layers,
            n_node=self.max_size,
            in_dim=self.node_dim,
            hid_dim=args.hid_dim,
            inv_rotate=args.inv_rotate,
            condition=args.condition
        )

    def forward(self, x, adj):
        """
        :param x: (B,n,10)
        :param adj: (B,4,n,n) dequantized
        :param adj_normalized: (B,4,n,n) normalized
        :return:
        """
        batch_size = x.shape[0]

        if self.args.deq_type == 'random':
            x_deq = x + self.args.deq_scale * torch.rand_like(x)
            adj_deq = adj + self.args.deq_scale * torch.rand_like(adj)

        z_x, sum_log_det_jacs_x = self.atom_model(x_deq, adj.clone())
        z_adj, sum_log_det_jacs_adj = self.bond_model(adj_deq)

        if isinstance(z_adj, torch.Tensor):
            z_adj = z_adj.view(batch_size, -1)

        out = [z_x, z_adj]
        logdet = [sum_log_det_jacs_x, sum_log_det_jacs_adj]

        return out, logdet, self.prior_ln_var

    def reverse(self, z, true_adj=None, grad=False):
        """
        Returns a molecule, given its latent vector.
        :param z_x : list of latent vectors. Length: num_trans
               z_adj : latent vector. Shape: [B, b*n*n]
            B = Batch size, n = number of atoms, b = number of bond types,
            d = number of atom types (Carbon, Oxygen etc.)
        :param true_adj: used for testing. An adjacency matrix of a real molecule
        :return: adjacency matrix and feature matrix of a molecule
        """
        z_x, z_adj = z
        batch_size = z_x[0].shape[0]
        with torch.no_grad():
            if true_adj is None:
                if isinstance(z_adj, torch.Tensor):
                    z_adj = z_adj.reshape(
                        batch_size, self.bond_dim, self.max_size, self.max_size)  # (B,b,n,n)
                h_adj = self.bond_model.reverse(z_adj)

                adj = (h_adj + h_adj.permute(0, 1, 3, 2))/2  # symmetry

                adj = adj.softmax(dim=1)
                max_bond = adj.max(dim=1).values.reshape(
                    batch_size, -1, self.max_size, self.max_size)  # (100,1,9,9)
                # (100,4,9,9) /  (100,1,9,9) -->  (100,4,9,9)
                adj = torch.floor(adj / max_bond)
            else:
                adj = true_adj

            x = self.atom_model.reverse(z_x, adj.clone())

            x = x.softmax(dim=2)
            max_atom = x.max(dim=2).values.reshape(
                batch_size, self.max_size, -1)
            x = torch.floor(x/max_atom)
        return x, adj

    def generate(self, num, temperature):
        prior_dist = torch.distributions.normal.Normal(torch.zeros(
            [self.latent_node_length+self.latent_edge_length]), temperature*torch.sqrt(torch.exp(self.prior_ln_var.cpu()))*torch.ones([self.latent_node_length+self.latent_edge_length]))
        z = prior_dist.sample((num,)).to(self.device)

        z = self.to_molecule_format(z)
        x, adj = self.reverse(z)

        return x, adj
    
    def resampling(self, x, adj, temperature, mode=0):
        bt_size = x.shape[0]
        results = [[x, adj]]
        with torch.no_grad():
            out, _, _ = self.forward(x, adj)
            z_x, z_adj = out
            x_new, adj_new = deepcopy(z_x), deepcopy(z_adj)
            for i in range(len(z_x)):
                if mode == 0:
                    x_new, adj_new = deepcopy(z_x), deepcopy(z_adj)
                x_nelement = x_new[i].nelement()//bt_size
                x_prior_dist = torch.distributions.normal.Normal(torch.zeros(
                    [x_nelement]), temperature*torch.ones([x_nelement]))
                x_part = x_prior_dist.sample((bt_size,)).to(self.device)
                x_new[i] = x_part.view(*x_new[i].shape)
                if i != 0:
                    adj_nelement = adj_new[i-1].nelement()//bt_size
                    adj_prior_dist = torch.distributions.normal.Normal(torch.zeros(
                            [adj_nelement]), temperature*torch.ones([adj_nelement]))
                    adj_part = adj_prior_dist.sample((bt_size,)).to(self.device)
                    adj_new[i-1] = adj_part.view(*adj_new[i-1].shape)
                x_cur, adj_cur = self.reverse([x_new, adj_new])
                results.append([x_cur, adj_cur])
        return results

    def to_molecule_format(self, z):
        num = z.shape[0]
        z_x, z_adj = z[:, :self.latent_node_length], z[:, self.latent_node_length:]
        z_xs = []
        n_node = self.max_size
        for i in range(self.atom_model.n_block-1):
            z_new, z_x = z_x.chunk(2, 1)
            z_xs.append(z_new.reshape(num, n_node, -1))
            n_node //= self.args.squeeze_fold
        z_xs.append(z_x.reshape(num, n_node, -1))

        # sample z_adj
        z_adjs = []
        n_node = self.max_size
        for i in range(self.bond_model.n_block-1):
            n_node //= self.bond_model.squeeze_fold
            z_adj = z_adj.view(num, -1, n_node, n_node)
            z_new, z_adj = z_adj.chunk(2, 1)
            z_adjs.append(z_new)
        z_adjs.append(z_adj.reshape(num, -1, n_node //
                      self.bond_model.squeeze_fold, n_node//self.bond_model.squeeze_fold))
        return [z_xs, z_adjs]

    def to_latent_format(self, z):
        batch_size = z[0][0].shape[0]
        z_x = torch.cat([i.reshape(batch_size, -1) for i in z[0]], dim=1)

        if isinstance(z[1], list):
            z_adj = torch.cat([i.reshape(batch_size, -1) for i in z[1]], dim=1)
        else:
            z_adj = z[1]
        
        h = torch.cat([z_x, z_adj], dim=1)
        return h

    def log_prob(self, z, logdet):
        # calculate probability of a region from probability density, minus constant has no effect on optimization
        logdet[0] = logdet[0] - self.latent_node_length * math.log(2.)
        # calculate probability of a region from probability density, minus constant has no effect on optimization
        logdet[1] = logdet[1] - self.latent_edge_length * math.log(2.)

        batch_size = z[0][0].shape[0]
        z_x = torch.cat([i.reshape(batch_size, -1) for i in z[0]], dim=1)

        if isinstance(z[1], list):
            z_adj = torch.cat([i.reshape(batch_size, -1) for i in z[1]], dim=1)
        else:
            z_adj = z[1]
        
        ll_node = -1/2 * (math.log(2 * math.pi) +
                          self.prior_ln_var + torch.exp(-self.prior_ln_var) * (z_x**2))
        ll_node = ll_node.sum(-1)  # (B)

        ll_edge = -1/2 * (math.log(2 * math.pi) +
                          self.prior_ln_var + torch.exp(-self.prior_ln_var) * (z_adj**2))
        ll_edge = ll_edge.sum(-1)  # (B)

        
        ll_node += logdet[0]  # ([B])
        ll_edge += logdet[1]  # ([B])

        return -torch.mean(ll_node)/(self.latent_node_length*math.log(2.)), -torch.mean(ll_edge)/(self.latent_edge_length*math.log(2.))

