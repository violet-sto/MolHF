import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

import torch
import numpy as np
import json
import networkx as nx
from tqdm import tqdm
from time import time, ctime
from datetime import datetime
from utils import construct_mol, correct_mol, set_random_seed
from envs import environment as env
from models.graphflow import squeeze_adj
from models.MolHF import MolHF
from rdkit import Chem
from torch.utils.data import DataLoader
from dataloader import PretrainDataset
import argparse

def arg_parse():
    parser = argparse.ArgumentParser(description='MolHF')

    # ******data args******
    parser.add_argument('--dataset', type=str,
                        default='zinc250k', help='name of dataset')
    parser.add_argument('--order', type=str, default='bfs',
                        help='order of atoms')
    parser.add_argument('--num_data', type=int,
                        default=None, help='num of data to train')
    parser.add_argument('--batch_size', type=int,
                        default=256, help='batch_size.')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='num works to generate data.')
    parser.add_argument('--seed', type=int, default=42, help='random seed')

    # ******model args******
    parser.add_argument('--model', type=str,
                        default='MolHF', help='name of model [MolHF]')
    parser.add_argument('--deq_type', type=str,
                        default='random', help='dequantization methods.')
    parser.add_argument('--deq_scale', type=float, default=0.6,
                        help='dequantization scale.(only for deq_type random)')
    parser.add_argument('--squeeze_fold', type=int, default=2,
                        help='squeeze fold')
    parser.add_argument('--n_block', type=int, default=4,
                        help='num block')
    parser.add_argument('--condition', action='store_true', default=False,
                        help='latent variables on condition')

    # ***atom model***
    parser.add_argument('--a_num_flows', type=int, default=6,
                        help='num of flows in RGBlock')
    parser.add_argument('--num_layers', type=int, default=2,
                        help='num of R-GCN layer in GraphAffineCoupling')
    parser.add_argument('--hid_dim', type=int, default=256,
                        help='hidden dim of R-GCN layer')
    parser.add_argument('--inv_rotate', action='store_true',
                        default=False, help='whether rotate node feature')
    # ***bond model***
    parser.add_argument('--b_num_flows', type=int, default=3,
                        help='num of flows in bond model')
    parser.add_argument('--filter_size', type=int, default=256,
                        help='num of filter size in AffineCoupling')
    parser.add_argument('--inv_conv', action='store_true',
                        default=False, help='whether use 1*1 conv')

    # ******optimization args******
    parser.add_argument('--train', action='store_true',
                        default=False, help='do training.')
    parser.add_argument('--save', action='store_true',
                        default=False, help='Save model.')
    parser.add_argument('--resample', action='store_true',
                        default=False, help='do resampling.')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Disables CUDA training.')
    parser.add_argument('--learn_prior', action='store_true',
                        default=False, help='learn log-var of gaussian prior.')
    parser.add_argument('--epochs', type=int, default=200,
                        help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Initial learning rate.')
    parser.add_argument('--ratio', type=float, default=1,
                        help='ratio for loss for GAN.')
    parser.add_argument('--weight_clip_value', type=float, default=0.01,
                        help='weight clip value for W-GAN')
    parser.add_argument("--lr_decay", type=float,
                        default=0.999995, help='learning rate decay')
    parser.add_argument('--init_checkpoint', type=str, default=None,
                        help='initialize from a checkpoint, if None, do not restore')
    parser.add_argument('--show_loss_step', type=int, default=20)

    # ******generation args******
    parser.add_argument('--gen_num', type=int, default=10000,
                        help='Number of generated molecules')
    parser.add_argument('--temperature', type=float, default=0.6,
                        help='temperature for normal distribution')
    parser.add_argument('--min_atoms', type=int, default=5,
                        help='minimum #atoms of generated mol, otherwise the mol is simply discarded')

    return parser.parse_args()


class Trainer:
    def __init__(self, train_loader, val_loader, args):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.args = args
        self.data_config = train_loader.dataset.data_config
        self.all_train_smiles = train_loader.dataset.all_smiles

        self.device = args.device

        self._model = MolHF(self.data_config, args).to(self.device)

        self._optimizer = torch.optim.Adam(
            self._model.parameters(), lr=self.args.lr)

        self.best_metric = -1
        self.start_epoch = 0
        self.Lambda = 10

    def save_model(self, var_list):
        args = self.args
        argparse_dict = vars(args)

        # Save hyperparameters
        with open(os.path.join(args.save_path, 'config.json'), 'w') as f:
            json.dump(argparse_dict, f, indent=4)

        latest_save_path = os.path.join(args.save_path, 'checkpoint.pth')

        torch.save({
            **var_list,
            'model_state_dict': self._model.state_dict(),
            'optimizer_state_dict': self._optimizer.state_dict()},
            latest_save_path
        )

    def initialize_from_checkpoint(self, train=True):
        checkpoint = torch.load(
            self.args.init_checkpoint, map_location=self.device)
        self._model.load_state_dict(
            checkpoint['model_state_dict'])

        if train:
            self._optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.best_metric = checkpoint['best_metric']
            self.start_epoch = checkpoint['cur_epoch'] + 1
        print('initialize from %s Done!' % self.args.init_checkpoint)

    def fit(self, mol_out_dir):
        t_total = time()
        best_metric = self.best_metric
        start_epoch = self.start_epoch

        metrics = {'total_loss': [], 'all_valid_rate': [], 'all_valid_without_check_rate': [
        ], 'all_unique_rate': [], 'all_novelty_rate': [], 'all_connectivity_rate': [], 'all_reconstruct_error': []}

        print('start fitting.')
        for epoch in range(self.args.epochs):
            start = time()
            epoch_loss, node_loss, edge_loss = self.train_epoch(
                epoch + start_epoch)
            print('Epoch {}: fitting done! Time {:.2f} seconds, Data: {}'.format(
                epoch, time() - start, ctime()))
            metrics['total_loss'].append(
                (epoch_loss, node_loss, edge_loss))

            mol_save_path = os.path.join(mol_out_dir, 'epoch%d.txt' % (
                epoch + start_epoch)) if mol_out_dir is not None else None
            cur_connectivity, cur_valid, cur_valid_without_check, cur_unique, cur_novelty, reconstruct_error, _ = self.generate_molecule(num=self.args.gen_num, epoch=epoch + start_epoch,
                                                                                                                                         out_path=mol_save_path, mute=True)
            print("dataset:{}, squeeze_fold:{}, n_block:{}, a_num_flows:{}, num_layers:{}, hid_dim:{}, b_num_flows:{}, filter_size:{}, num_data:{}, lr:{}, ratio:{}"
                  .format(self.args.dataset, self.args.squeeze_fold, self.args.n_block, self.args.a_num_flows, self.args.num_layers, self.args.hid_dim, self.args.b_num_flows,
                          self.args.filter_size, self.args.num_data, self.args.lr, self.args.ratio))
            metrics['all_valid_rate'].append(cur_valid)
            metrics['all_valid_without_check_rate'].append(
                cur_valid_without_check)
            metrics['all_unique_rate'].append(cur_unique)
            metrics['all_novelty_rate'].append(cur_novelty)
            metrics['all_connectivity_rate'].append(cur_connectivity)
            metrics['all_reconstruct_error'].append(reconstruct_error)
            if self.args.save:
                print('saving metrics...')
                with open(os.path.join(args.save_path, 'metrics.json'), 'w') as f:
                    json.dump(metrics, f, indent=4)

            if cur_valid_without_check > best_metric:
                best_metric = cur_valid_without_check
                if self.args.save:
                    var_list = {'cur_epoch': epoch + start_epoch,
                                'best_metric': best_metric, }
                    self.save_model(var_list)

        print("Optimization Finished!")
        print("Total time elapsed: {:.4f}s".format(time() - t_total))

    def train_epoch(self, epoch_cnt):
        t_start = time()
        batch_losses = []
        node_losses = []
        edge_losses = []
        self._model.train()
        for idx, batch_data in enumerate(tqdm(self.train_loader, desc='Iteration')):
            batch_time_s = time()
            x = batch_data['node'].to(self.device)  # (B, N, 10)
            adj = batch_data['adj'].to(self.device)  # (B, 4, N, N)

            if self.args.deq_type == 'random':
                out_z, out_logdet, ln_var = self._model(
                    x, adj)
                loss_node, loss_edge = self._model.log_prob(out_z, out_logdet)
                loss = loss_node + loss_edge
                # TODO: add mask for different molecule size, i.e. do not model the distribution over padding nodes.

            elif self.args.deq_type == 'variational':
                out_z, out_logdet, out_deq_logp, out_deq_logdet = self._model(
                    x, adj)
                ll_node, ll_edge, ll_deq_node, ll_deq_edge = self._model.log_prob(
                    out_z, out_logdet, out_deq_logp, out_deq_logdet)
                loss = -1. * ((ll_node-ll_deq_node) + (ll_edge-ll_deq_edge))
            else:
                raise ValueError(
                    'unsupported dequantization method: (%s)' % self.deq_type)


            self._optimizer.zero_grad()
            loss.backward()
            self._optimizer.step()

            batch_losses.append(loss.item())
            node_losses.append(loss_node.item())
            edge_losses.append(loss_edge.item())

            if idx % self.args.show_loss_step == 0 or (epoch_cnt == 0 and idx <= 100):
                print('Epoch: {} | step: {} | time: {:.5f} | loss: {:.5f} | loss_node: {:.5f} | loss_edge:{:.5f} | ln_var: {:.5f}'.format(
                    epoch_cnt, idx, time() - batch_time_s, batch_losses[-1], loss_node.item(), loss_edge.item(), ln_var.item()))
        epoch_loss = sum(batch_losses) / len(batch_losses)
        node_loss = sum(node_losses) / len(node_losses)
        edge_loss = sum(edge_losses) / len(edge_losses)

        print('Epoch: {}, loss {:.5f}, epoch time {:.5f}'.format(
            epoch_cnt, epoch_loss, time()-t_start))
        return epoch_loss, node_loss, edge_loss

    def generate_molecule(self, num=100, epoch=None, out_path=None, mute=False, correct_validity=False, save_good_mol=False, min_atoms=5):
        generate_start_t = time()

        self._model.eval()
        connected_adjs = []
        all_smiles = []
        pure_valid_smiles = []
        valid_smiles = []

        xs = []
        adjs = []
        for i in range(num//100):
            x, adj = self._model.generate(
                100, self.args.temperature)
            xs.append(x)
            adjs.append(adj)

        xs = torch.cat(xs, dim=0)
        adjs = torch.cat(adjs, dim=0)

        for x, adj in zip(xs, adjs):
            try:
                connected_adjs.append(nx.is_connected(
                    nx.from_numpy_matrix(adj[:3].sum(0)[adj[:3].sum((0, 1)) != 0][:, adj[:3].sum((0, 1)) != 0].cpu().numpy())))
            except:
                connected_adjs.append(False)

            mol, smiles = construct_mol(x, adj, num2atom, atom_valency)
            all_smiles.append(smiles)

            if smiles != '' and env.check_chemical_validity(mol) and mol.GetNumAtoms() >= self.args.min_atoms:
                pure_valid_smiles.append(smiles)
                valid_smiles.append(smiles)
            else:    
                if correct_validity:
                    cmol = correct_mol(mol)
                    vcmol = env.check_chemical_validity_with_seg(
                        cmol, largest_connected_comp=True)
                    valid_smiles.append(vcmol[1])

        if out_path is not None and self.args.save:
            valid_smiles = sorted(valid_smiles, key=len)
            with open(out_path, 'w') as f:
                cnt = 0
                for i in range(len(valid_smiles)):
                    num_atom = Chem.MolFromSmiles(
                        valid_smiles[i]).GetNumAtoms()
                    f.write(valid_smiles[i] + ", " + str(num_atom) + '\n')
                    cnt += 1
            print('writing %d smiles into %s done!' % (cnt, out_path))

        print("Original_smiles:", all_smiles[:5])
        for idx, s in enumerate(valid_smiles):
            print('[{}] {}'.format(idx+1, s))

        # The percentage of valid adjacent matrix among all the generated graphs
        Connectivity = 100*sum(connected_adjs)/num

        # The percentage of valid molecules among all the generated graphs
        Validity = 100*len(valid_smiles)/num
        Validity_without_check = 100*len(pure_valid_smiles)/num

        # The percentage of unique molecules among all the generated valid molecules.
        unique_smiles = list(set(valid_smiles))
        if len(valid_smiles) != 0:
            Uniqueness = 100*len(unique_smiles)/len(valid_smiles)
        else:
            Uniqueness = 0

        # The percentage of generated valid molecules not appearing in training set.
        # Only include unique smiles.
        valid_smiles = unique_smiles
        Novelty = 0
        for smiles in valid_smiles:
            if mol not in self.all_train_smiles:
                Novelty += 1

        if len(valid_smiles) != 0:
            Novelty = 100*Novelty/len(valid_smiles)
        else:
            Novelty = 0

        mol_atom_size = [Chem.MolFromSmiles(x).GetNumAtoms() for x in valid_smiles]

        # The percentage of the molecules that can be reconstructed from latent vectors.
        sampled_data = next(iter(self.train_loader))
        x_origin = sampled_data['node'].to(self.device)
        adj_origin = sampled_data['adj'].to(self.device)

        print('--------Distribution of the real molecules:')
        squeezed_adj = squeeze_adj(adj_origin[:, :3], 8)
        print("Total num of edges: {}".format(squeezed_adj.sum()))
        squeezed_adj[:, :, range(5), range(5)] = 0
        print("num of edges in non-diagonal block: {}".format(squeezed_adj.sum()))

        with torch.no_grad():
            out_z, out_logdet, ln_var = self._model(x_origin, adj_origin)
            x_reconstruct, adj_reconstruct = self._model.reverse(
                out_z)

            reconstruct_error = (torch.abs(
                x_reconstruct-x_origin).sum().item(), torch.abs(adj_reconstruct-adj_origin).sum().item())
            if sum(reconstruct_error) != 0:
                print("Irreversible! reconstruct loss: {} (x) and {} (adj)".format(
                    reconstruct_error[0], reconstruct_error[1]))
            else:
                print('100% reconstruct!')

        print('Time of generating {} molecules: {:.5f} at epoch:{} | Connectivity: {:.5f} | valid rate: {:.5f} | valid w/o check rage: {:.5f} | unique rate: {:.5f} | novelty: {:.5f}'.format(
            num, time()-generate_start_t, epoch, Connectivity, Validity, Validity_without_check, Uniqueness, Novelty))


        return Connectivity, Validity, Validity_without_check, Uniqueness, Novelty, reconstruct_error, mol_atom_size

    
    def resampling_molecules(self, resample_mode=0):
        sampled_data = next(iter(self.train_loader))
        xs = sampled_data['node'].to(self.device)
        adjs = sampled_data['adj'].to(self.device)

        samples = self._model.resampling(xs, adjs, self.args.temperature, resample_mode)
        logs = []
        for sample in samples:
            log = []
            xs_cur, adjs_cur = sample
            for x, adj in zip(xs_cur, adjs_cur):
                mol, smiles = construct_mol(x, adj, num2atom, atom_valency)
                if smiles != '' and env.check_chemical_validity(mol):
                    log.append(smiles)
                else:
                    log.append("incorrect molecule")
            logs.append(log)
        with open("./{}_resampling_molecules_resample_mode_{}.txt".format(self.args.dataset, resample_mode), 'w') as f:
            cnt = 0
            for i, cur_smiles in enumerate(zip(*logs)):
                f.write("[{}]:{}\n".format(i, ",".join(cur_smiles)))
                cnt += 1
        print('writing %d smiles into %s done!' % (cnt, logs))


if __name__ == '__main__':
    args = arg_parse()
    set_random_seed(args.seed)
    if args.save:
        dt = datetime.now()
        # TODO: Add more information.
        log_dir = os.path.join('./save_pretrain', args.model, args.order, '{}_{:02d}-{:02d}-{:02d}'.format(
            dt.date(), dt.hour, dt.minute, dt.second))
        args.save_path = log_dir

        if not os.path.exists(args.save_path):
            os.makedirs(args.save_path)

    if args.dataset == 'polymer':
        # polymer
        num2atom = {0: 6, 1: 7, 2: 8, 3: 9, 4: 14, 5: 15, 6: 16}
        atom_valency = {6: 4, 7: 3, 8: 2, 9: 1, 14: 4, 15: 3, 16: 2}
    else:
        # zinc250k
        num2atom = {0: 6, 1: 7, 2: 8, 3: 9, 4: 15, 5: 16, 6: 17, 7: 35, 8: 53}
        atom_valency = {6: 4, 7: 3, 8: 2, 9: 1,
                        15: 3, 16: 2, 17: 1, 35: 1, 53: 1}

    # load data
    data_path = os.path.join('./data_preprocessed', args.dataset)
    with open(os.path.join(data_path, 'config.txt'), 'r') as f:
        data_config = eval(f.read())
    dataset = PretrainDataset(
        data_path, data_config, args)
    train_loader = DataLoader(dataset, batch_size=args.batch_size,
                              collate_fn=PretrainDataset.collate_fn, shuffle=True, num_workers=args.num_workers, drop_last=True)

    trainer = Trainer(train_loader, None, args)
    if args.init_checkpoint is not None:
        trainer.initialize_from_checkpoint(train=args.train)
    if args.train:
        if args.save:
            mol_out_dir = os.path.join(log_dir, 'mols')

            if not os.path.exists(mol_out_dir):
                os.makedirs(mol_out_dir)
        else:
            mol_out_dir = None
        start = time()
        trainer.fit(mol_out_dir=mol_out_dir)
        print('Task model fitting done! Time {:.2f} seconds, Data: {}'.format(
            time() - start, ctime()))

    elif args.resample:
        trainer.resampling_molecules(resample_mode=0)
    else:
        print('Start generating!')
        start = time()
        valid_ratio = []
        unique_ratio = []
        novel_ratio = []
        valid_5atom_ratio = []
        valid_39atom_ratio = []
        for i in range(5):
            _, Validity, Validity_without_check, Uniqueness, Novelty, _, mol_atom_size = trainer.generate_molecule(
                args.gen_num)
            valid_ratio.append(Validity)
            unique_ratio.append(Uniqueness)
            novel_ratio.append(Novelty)
            valid_5atom_ratio.append(
                np.sum(np.array(mol_atom_size) >= 5) / args.gen_num * 100)
            valid_39atom_ratio.append(
                np.sum(np.array(mol_atom_size) >= 39) / args.gen_num * 100)

        print("validity: mean={:.2f}%, sd={:.2f}%, vals={}".format(
            np.mean(valid_ratio), np.std(valid_ratio), valid_ratio))
        print("validity if atom >= 5: mean={:.2f}%, sd={:.2f}%, vals={}".format(
            np.mean(valid_5atom_ratio), np.std(valid_5atom_ratio), valid_5atom_ratio))
        print("validity if atom >= 39: mean={:.2f}%, sd={:.2f}%, vals={}".format(
            np.mean(valid_39atom_ratio), np.std(valid_39atom_ratio), valid_39atom_ratio))
        print("novelty: mean={:.2f}%, sd={:.2f}%, vals={}".format(
            np.mean(novel_ratio), np.std(novel_ratio), novel_ratio))
        print("uniqueness: mean={:.2f}%, sd={:.2f}%, vals={}".format(
            np.mean(unique_ratio), np.std(unique_ratio), unique_ratio))
        print('Task random generation done! Time {:.2f} seconds, Data: {}'.format(
            time() - start, ctime()))
