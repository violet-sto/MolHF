import re
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from rdkit import Chem
from envs import environment as env
import re
import random

from preprocess import SmilesPreprocessor
num2bond = {0: Chem.rdchem.BondType.SINGLE,
            1: Chem.rdchem.BondType.DOUBLE, 2: Chem.rdchem.BondType.TRIPLE}


def rescale_adj(adj, num_nodes=None, improved=False, add_self_loops=False, type='all'):
    # Previous paper didn't use rescale_adj.
    # In their implementation, the normalization sum is: num_neighbors = F.sum(adj, axis=(1, 2))
    # In this implementation, the normaliztion term is different
    # raise NotImplementedError
    # (256,4,9, 9):
    # 4: single, double, triple, and bond between disconnected atoms (negative mask of sum of previous)
    # 1-adj[i,:3,:,:].sum(dim=0) == adj[i,4,:,:]
    # usually first 3 matrices have no diagnal, the last has.
    # A_prime = self.A + sp.eye(self.A.shape[0])

    adj = adj.clone()  # !keep raw adj stable

    if num_nodes is None:
        num_nodes = adj.shape[-1]

    if add_self_loops:
        fill_value = 2. if improved else 1.
        adj += (torch.eye(num_nodes)*fill_value).to(adj.device)
    else:
        adj[:, :, range(num_nodes), range(num_nodes)] = 0

    # TODO D^{-1/2}*A*D^{-1/2}

    if type == 'view':
        out_degree = adj.sum(dim=-1)
        out_degree_sqrt_inv = out_degree.pow(-1)
        out_degree_sqrt_inv[out_degree_sqrt_inv == float('inf')] = 0
        # (256,4,9,1) * (256, 4, 9, 9) = (256, 4, 9, 9)
        adj_prime = out_degree_sqrt_inv.unsqueeze(-1) * adj
    else:  # default type all
        num_neighbors = adj.sum(dim=(1, 2)).float()
        num_neighbors_inv = num_neighbors.pow(-1)
        num_neighbors_inv[num_neighbors_inv == float('inf')] = 0
        adj_prime = num_neighbors_inv[:, None, None, :] * adj

    return adj_prime


def mask_adj(adj, stride):
    dis_mask = torch.eye(adj.shape[-1]//stride).to(adj.device)
    dis_mask = dis_mask.repeat_interleave(stride, dim=0)
    dis_mask = dis_mask.repeat_interleave(stride, dim=1)
    permute = list(range(stride//2, len(dis_mask)))+list(range(stride//2))
    dec_mask = dis_mask[permute][:, permute]

    return adj*dis_mask, adj*dec_mask


def smiles_to_adj(mol_smiles, data_name='zinc250k'):

    if data_name == 'zinc250k':
        preprocessor = SmilesPreprocessor(
            add_Hs=False, kekulize=True, max_atoms=38, max_size=40)
    elif data_name == 'qm9':
        preprocessor = SmilesPreprocessor(
            add_Hs=False, kekulize=True, max_atoms=9, max_size=9)

    mol, canonical_smiles = preprocessor._prepare_mol(
        mol_smiles)  # newly added crucial important!!!
    atoms, adj, mol_size = preprocessor._get_features(mol)
    return atoms, adj[:3]


def check_valency(mol):
    """
    Checks that no atoms in the mol have exceeded their possible
    valency
    :return: True if no valency issues, False otherwise
    """
    try:
        Chem.SanitizeMol(
            mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_PROPERTIES)
        return True, None
    except ValueError as e:
        e = str(e)
        p = e.find('#')
        e_sub = e[p:]
        atomid_valence = list(map(int, re.findall(r'\d+', e_sub)))
        return False, atomid_valence


def construct_mol(x, adj, num2atom, atom_valency):
    mol = Chem.RWMol()
    atoms = torch.argmax(x, axis=1)
    atoms_exist = atoms != len(num2atom)
    atoms = atoms[atoms_exist]
    for atom in atoms:
        mol.AddAtom(Chem.Atom(num2atom[atom.item()]))

    # A (edge_type, num_node, num_node)
    adj = torch.argmax(adj, axis=0)
    adj = adj[atoms_exist, :][:, atoms_exist]
    adj[adj == 3] = -1
    for start, end in torch.nonzero(adj+1):
        if start > end:
            mol.AddBond(start.item(), end.item(),
                        num2bond[adj[start, end].item()])
            # add formal charge to atom: e.g. [O+], [N+] [S+]
            # not support [O-], [N-] [S-]  [NH+] etc.
            flag, atomid_valence = check_valency(mol)
            if flag:
                continue
            else:
                assert len(atomid_valence) == 2
                idx = atomid_valence[0]
                v = atomid_valence[1]
                an = mol.GetAtomWithIdx(idx).GetAtomicNum()
                if an in (7, 8, 16) and (v - atom_valency[an]) == 1:
                    mol.GetAtomWithIdx(idx).SetFormalCharge(1)
    try:
        smiles = Chem.MolToSmiles(mol, isomericSmiles=True)
    except ValueError:
        smiles = ""

    return mol, smiles


def correct_mol(mol):
    # xsm = Chem.MolToSmiles(x, isomericSmiles=True)
    while True:
        flag, atomid_valence = check_valency(mol)
        if flag:
            break
        else:
            assert len(atomid_valence) == 2
            idx = atomid_valence[0]
            v = atomid_valence[1]
            queue = []
            for b in mol.GetAtomWithIdx(idx).GetBonds():
                queue.append(
                    (b.GetIdx(), int(b.GetBondType())-1,
                     b.GetBeginAtomIdx(), b.GetEndAtomIdx())
                )
            queue.sort(key=lambda tup: tup[1], reverse=True)
            if len(queue) > 0:
                start = queue[0][2]
                end = queue[0][3]
                t = queue[0][1]
                mol.RemoveBond(start, end)
                if t >= 1:
                    mol.AddBond(start, end, num2bond[t-1])
                # if '.' in Chem.MolToSmiles(mol, isomericSmiles=True):
                #     print(tt)
                #     print(Chem.MolToSmiles(mol, isomericSmiles=True))

    return mol


def valid_mol(x):
    s = Chem.MolFromSmiles(Chem.MolToSmiles(
        x, isomericSmiles=True)) if x is not None else None
    if s is not None and '.' not in Chem.MolToSmiles(s, isomericSmiles=True):
        return s
    return None


def valid_mol_can_with_seg(x, largest_connected_comp=True):
    # mol = None
    if x is None:
        return None
    sm = Chem.MolToSmiles(x, isomericSmiles=True)
    mol = Chem.MolFromSmiles(sm)
    if largest_connected_comp and '.' in sm:
        # 'C.CC.CCc1ccc(N)cc1CCC=O'.split('.')
        vsm = [(s, len(s)) for s in sm.split('.')]
        vsm.sort(key=lambda tup: tup[1], reverse=True)
        mol = Chem.MolFromSmiles(vsm[0][0])
    return mol


def check_validity(x, adj, num2atom, atom_valency, gpu=-1, return_unique=True,
                   correct_validity=True, largest_connected_comp=True, debug=True):
    """

    :param adj:  (100,4,9,9)
    :param x: (100.9,5)
    :param atomic_num_list: [6,7,8,9,0]
    :param gpu:  e.g. gpu0
    :param return_unique:
    :return:
    """
    # adj = _to_numpy_array(adj)  # , gpu)  (1000,4,9,9)
    # x = _to_numpy_array(x)  # , gpu)  (1000,9,5)
    if correct_validity:
        # valid = [valid_mol_can_with_seg(construct_mol_with_validation(x_elem, adj_elem, atomic_num_list)) # valid_mol_can_with_seg
        #          for x_elem, adj_elem in zip(x, adj)]
        valid = []
        for x_elem, adj_elem in zip(x, adj):
            mol = construct_mol(x_elem, adj_elem, num2atom, atom_valency)[0]
            # Chem.Kekulize(mol, clearAromaticFlags=True)
            cmol = correct_mol(mol)
            # valid_mol_can_with_seg(cmol)  # valid_mol(cmol)  # valid_mol_can_with_seg
            vcmol = valid_mol_can_with_seg(
                cmol, largest_connected_comp=largest_connected_comp)
            # Chem.Kekulize(vcmol, clearAromaticFlags=True)
            valid.append(vcmol)
    else:
        valid = [valid_mol(construct_mol(x_elem, adj_elem)[0])
                 for x_elem, adj_elem in zip(x, adj)]  # len()=1000
    # len()=valid number, say 794
    valid = [mol for mol in valid if mol is not None]
    if debug:
        print("valid molecules: {}/{}".format(len(valid), adj.shape[0]))
        for i, mol in enumerate(valid):
            print("[{}] {}".format(i, Chem.MolToSmiles(mol, isomericSmiles=False)))

    n_mols = x.shape[0]
    valid_ratio = len(valid)/n_mols  # say 794/1000
    valid_smiles = [Chem.MolToSmiles(
        mol, isomericSmiles=False) for mol in valid]
    unique_smiles = list(set(valid_smiles))  # unique valid, say 788
    unique_ratio = 0.
    if len(valid) > 0:
        unique_ratio = len(unique_smiles)/len(valid)  # say 788/794
    if return_unique:
        valid_smiles = unique_smiles
    valid_mols = [Chem.MolFromSmiles(s) for s in valid_smiles]
    abs_unique_ratio = len(unique_smiles)/n_mols
    if debug:
        print("valid: {:.3f}%, unique: {:.3f}%, abs unique: {:.3f}%".
              format(valid_ratio * 100, unique_ratio * 100, abs_unique_ratio * 100))
    results = dict()
    results['valid_mols'] = valid_mols
    results['valid_smiles'] = valid_smiles
    results['valid_ratio'] = valid_ratio*100
    results['unique_ratio'] = unique_ratio*100
    results['abs_unique_ratio'] = abs_unique_ratio * 100

    return results


def save_mol_png(mol, filepath, size=(600, 600)):
    Chem.Draw.MolToFile(mol, filepath, size=size)


def adj_to_smiles(atoms, adj, num2atom, atom_valency):
    # adj = _to_numpy_array(adj, gpu)
    # x = _to_numpy_array(x, gpu)
    valid = [construct_mol(x_elem, adj_elem, num2atom, atom_valency)[1]
             for x_elem, adj_elem in zip(atoms, adj)]
    return valid


def set_random_seed(seed, deterministic=True):
    """Set random seed."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
