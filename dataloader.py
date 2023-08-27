import numpy as np
import os
import networkx as nx
import copy
import torch
from torch.utils.data import Dataset

class PretrainDataset(Dataset):
    def __init__(self, data_path, data_config, args, freedom=0):
        node_features, adj_features, mol_sizes, all_smiles= self.read_molecules(
            data_path)
        self.node_features = node_features
        self.adj_features = adj_features
        self.mol_sizes = mol_sizes
        self.all_smiles = all_smiles
        self.data_config = data_config
        self.data_config['max_size'] = self.data_config['max_size'] + freedom

        # C N O F P S Cl Br I virtual
        self.atom_list = data_config['atom_list']
        self.max_size = data_config['max_size']
        self.order = args.order
        print('Atom order: {}'.format(self.order))
        self.node_dim = len(self.atom_list)
        self.is_mol_property = False
        self._indices = list(range(args.num_data)) if args.num_data is not None else None

    def indices(self):
        return range(len(self.all_smiles)) if self._indices is None else self._indices

    def shuffle(self, return_perm: bool = False):
        indices = self.indices()
        perm = torch.randperm(len(self))
        dataset = copy.copy(self)
        dataset._indices = [indices[i] for i in perm]
        return (dataset, perm) if return_perm is True else dataset

    def __len__(self):
        return len(self.indices())

    def __getitem__(self, idx):
        idx = self.indices()[idx]
        node_feature_copy = self.node_features[idx].copy()  # (N)
        adj_feature_copy = self.adj_features[idx].copy().astype(np.float32)  # (3, N, N)

        node_feature, adj_feature = get_mol_data(node_feature_copy, adj_feature_copy, self.data_config)

        if self.is_mol_property:
            return {'node': torch.Tensor(node_feature), 'adj': torch.Tensor(adj_feature), 'property':torch.Tensor(self.mol_property[idx])}
        return {'node': torch.Tensor(node_feature), 'adj': torch.Tensor(adj_feature)}


    def read_molecules(self, path):
        print('reading data from %s' % path)
        node_features = np.load(os.path.join(path, 'node_features.npy'))
        adj_features = np.load(os.path.join(path, 'adj_features.npy'))
        mol_sizes = np.load(os.path.join(path, 'mol_sizes.npy'))
        smiles = np.load(os.path.join(path, 'smiles.npy'))

        return node_features, adj_features, mol_sizes, smiles

    @staticmethod
    def collate_fn(batch):
        batch = list(filter(lambda x: x is not None, batch))
        return torch.utils.data.dataloader.default_collate(batch)

def get_mol_data(node_feature_copy, adj_feature_copy, data_config, freedom=0):
    atom_list = data_config['atom_list']
    max_size = data_config['max_size']
    mol_size = (node_feature_copy != 0).sum()
    node_dim = len(atom_list)
    node_feature_copy = np.pad(node_feature_copy, ((0, freedom)))
    adj_feature_copy = np.pad(adj_feature_copy, ((0, 0), (0, freedom), (0, freedom)))
    
    ######## 1.Get permutation and bfs ######
    pure_adj = np.sum(adj_feature_copy, axis=0)[
        :mol_size, :mol_size]  # (mol_size, mol_size)
    local_perm = np.random.permutation(mol_size)  # (first perm graph)
    adj_perm = pure_adj[np.ix_(local_perm, local_perm)]
    adj_perm_matrix = np.asmatrix(adj_perm)
    G = nx.from_numpy_matrix(adj_perm_matrix)

    # get a bfs order of permed graph
    start_idx = np.random.randint(mol_size)
    bfs_perm = np.array(bfs_seq(G, start_idx))
    bfs_perm_origin = local_perm[bfs_perm]

    # 2.optimize order according to substructure
    perm = bfs_perm_origin
    perm = np.concatenate([perm, np.arange(mol_size, max_size)])


    ######## 3.Get node_feature and adj_feature #######
    node_feature_copy = node_feature_copy[np.ix_(perm)]  # (N)
    perm_index = np.ix_(perm, perm)
    for i in range(3):
        adj_feature_copy[i] = adj_feature_copy[i][perm_index]

    node_feature = np.zeros(
        (max_size, node_dim), dtype=np.float32)  # (N,10)
    for i in range(max_size):
        index = atom_list.index(node_feature_copy[i])
        node_feature[i, index] = 1.

    adj_feature = np.concatenate([adj_feature_copy, 1 - np.sum(
        adj_feature_copy, axis=0, keepdims=True)], axis=0).astype(np.float32)  # (4, N, N)
    return node_feature, adj_feature


def bfs_seq(G, start_id):
    '''
    get a bfs node sequence
    :param G:
    :param start_id:
    :return:
    '''
    dictionary = dict(nx.bfs_successors(G, start_id))
    start = [start_id]
    output = [start_id]
    while len(start) > 0:
        next = []
        while len(start) > 0:
            current = start.pop(0)
            neighbor = dictionary.get(current)
            if neighbor is not None:
                next = next + neighbor
        output = output + next
        start = next
    return output
