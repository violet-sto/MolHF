import os
import sys
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"

# for linux env.
sys.path.insert(0,'..')
import argparse
from distutils.util import strtobool
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from tqdm import tqdm
import networkx as nx
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from copy import deepcopy
# from mflow.generate import generate_mols_along_axis
from dataloader import PretrainDataset
from models.MolHF import MolHF
from torch.utils.data import DataLoader
from envs import environment as env
from envs.timereport import TimeReport
from envs.environment import penalized_logp, qed 
from utils import check_validity, adj_to_smiles, smiles_to_adj, construct_mol
from multiprocessing import Pool
from sklearn.metrics import r2_score, mean_absolute_error
from dataloader import get_mol_data
from time import time, ctime
import functools
print = functools.partial(print, flush=True)

class FlowProp(nn.Module):
    def __init__(self, model:MolHF, hidden_size):
        super(FlowProp, self).__init__()
        self.model = model
        self.latent_node_length = model.latent_node_length
        self.latent_edge_length = model.latent_edge_length
        self.latent_size = self.latent_node_length + self.latent_edge_length
        self.hidden_size = hidden_size

        vh = (self.latent_size,) + tuple(hidden_size) + (1,)
        modules = []
        for i in range(len(vh)-1):
            modules.append(nn.Linear(vh[i], vh[i+1]))
            if i < len(vh) - 2:
                modules.append(nn.Tanh())
        self.propNN = nn.Sequential(*modules)

    def encode(self, x, adj):
        z, logdet, _  = self.model(x, adj)  # z = [h, adj_h]
        return z, logdet

    def reverse(self, z):
        out = self.model.to_molecule_format(z)
        x, adj = self.model.reverse(out, true_adj=None)
        return x, adj

    def forward(self, x, adj):
        z, sum_log_det_jacs = self.encode(x, adj)
        h = self.model.to_latent_format(z)
        output = self.propNN(h)  # do I need to add nll of the unsupervised part? or just keep few epoch? see the results
        return output, z,  sum_log_det_jacs

def train_model(model, optimizer, train_loader, metrics, col, tr, epoch):
    log_step = 20
    train_iter_per_epoch = len(train_loader)
    
    print("Training...")
    model.train()
    total_pd_y = []
    total_true_y = []
    for i, batch in enumerate(train_loader):
        x = batch['node'].to(args.device)   # (bs,9,5)
        adj = batch['adj'].to(args.device)   # (bs,4,9, 9)
        true_y = batch['property'][:, col].unsqueeze(1).float().to(args.device)
        # model and loss
        optimizer.zero_grad()
        y, z, sum_log_det_jacs = model(x, adj)
        out_z, out_logdet, ln_var = model.model(
                x, adj)
        loss_node, loss_edge = model.model.log_prob(out_z, out_logdet)
        loss_mle = loss_node + loss_edge
        
        total_pd_y.append(y)
        total_true_y.append(true_y)
        loss_prop = metrics(y, true_y)
        
        loss = loss_mle+args.ratio*loss_prop
        
        loss.backward()
        optimizer.step()
        tr.update()
        
        # Print log info
        if (i + 1) % log_step == 0:  # i % args.log_step == 0:
            print('Epoch [{}/{}], Iter [{}/{}], loss: {:.5f}, loss_mle: {:.5f}, loss_prop: {:.5f}, {:.2f} sec/iter, {:.2f} iters/sec: '.
                    format(epoch + 1, args.max_epochs, i + 1, train_iter_per_epoch,
                            loss.item(), loss_mle.item(), loss_prop.item(),
                            tr.get_avg_time_per_iter(), tr.get_avg_iter_per_sec()))
            tr.print_summary()
    total_pd_y = torch.cat(total_pd_y, dim=-1)
    total_true_y = torch.cat(total_true_y, dim=-1)
    mse = metrics(total_pd_y, total_true_y)
    mae = mean_absolute_error(total_true_y.cpu().detach().numpy(), total_pd_y.cpu().detach().numpy())
    r2 = r2_score(total_true_y.cpu().detach().numpy(), total_pd_y.cpu().detach().numpy())
    print("Training, loss_mle:{}, loss_prop:{}, mse:{}, mae:{}, r2:{}".format(loss_mle.item(), loss_prop.item(), mse, mae, r2))
  
def validate_model(model, valid_loader, metrics, col, tr, epoch):
    log_step = 20
    valid_iter_per_epoch = len(valid_loader)
    
    print("Validating...")    
    model.eval()
    total_pd_y = []
    total_true_y = []
    with torch.no_grad():
        for i, batch in enumerate(valid_loader):
            x = batch['node'].to(args.device)   # (bs,9,5)
            adj = batch['adj'].to(args.device)   # (bs,4,9, 9)
            true_y = batch['property'][:, col].unsqueeze(1).float().to(args.device)
            # model and loss
            y, z, sum_log_det_jacs = model(x, adj)
            total_pd_y.append(y)
            total_true_y.append(true_y)
            loss_prop = metrics(y, true_y)
            tr.update()
            # Print log info
            if (i + 1) % log_step == 0:  # i % args.log_step == 0:
                print('Epoch [{}/{}], Iter [{}/{}], loss_prop: {:.5f}, {:.2f} sec/iter, {:.2f} iters/sec: '.
                        format(epoch + 1, args.max_epochs, i + 1, valid_iter_per_epoch,
                                loss_prop.item(),
                                tr.get_avg_time_per_iter(), tr.get_avg_iter_per_sec()))
                tr.print_summary()
        total_pd_y = torch.cat(total_pd_y, dim=-1)
        total_true_y = torch.cat(total_true_y, dim=-1)
        mse = metrics(total_pd_y, total_true_y)
        mae = mean_absolute_error(total_true_y.cpu().detach().numpy(), total_pd_y.cpu().detach().numpy())
        r2 = r2_score(total_true_y.cpu().detach().numpy(), total_pd_y.cpu().detach().numpy())
        print("Validating, loss_prop:{}, mse:{}, mae:{}, r2:{}".format(loss_prop.item(), mse, mae, r2))
        
    return r2   

def generate_molecule(model, train_loader, args, epoch, num=100):
    model.eval()
    connected_adjs = []
    all_smiles = []
    pure_valid_smiles = []
    valid_smiles = []
    start_t = time()

    xs = []
    adjs = []
    for i in range(num//100):
        x, adj = model.model.generate(100, 0.6)
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

        if smiles != '' and env.check_chemical_validity(mol) and mol.GetNumAtoms() >= 5:
            pure_valid_smiles.append(smiles)
            valid_smiles.append(smiles)

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

    print('Time of generating {} molecules: {:.5f} at epoch:{} | Connectivity: {:.5f} | valid rate: {:.5f} | valid w/o check rage: {:.5f} | unique rate: {:.5f}'.format(
        num, time()-start_t, epoch, Connectivity, Validity, Validity_without_check, Uniqueness))

def fit_model(model, train_loader, val_loader, args, property_model_path):
    start = time()
    print("Start at Time: {}".format(ctime()))
    # Loss and optimizer
    metrics = nn.MSELoss()
    best_metrics = float('-inf')
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # batch_size = data.batch_size
    train_iter_per_epoch = len(train_loader)
    valid_iter_per_epoch = len(val_loader)
    tr = TimeReport(total_iter = args.max_epochs * (train_iter_per_epoch+valid_iter_per_epoch))
    if args.property_name == 'qed':
        col = 0 # [0,1]
    elif args.property_name == 'plogp':
        col = 1  # unbounded, normalized later???
    else:
        raise ValueError("Wrong property_name{}".format(args.property_name))

    for epoch in range(args.max_epochs):
        print("In epoch {}, Time: {}".format(epoch + 1, ctime()))
        generate_molecule(model, train_loader, args, epoch)
        train_model(model, optimizer, train_loader, metrics, col, tr, epoch)
        cur_metrics = validate_model(model, valid_loader, metrics, col, tr, epoch)
        if best_metrics < cur_metrics:
            best_metrics = cur_metrics
            print("Epoch {}, saving {} regression model to: {}".format(epoch+1, args.property_name, property_model_path))
            torch.save(model.state_dict(), property_model_path)
        
    tr.print_summary()
    tr.end()
    
    print("[fit_model Ends], Start at {}, End at {}, Total {}".
          format(ctime(start), ctime(), time()-start))
    print('Train and save model done! Time {:.2f} seconds'.format(time() - start))
    return model


# ######### all latent variables opt ##########
def optimize_mol(property_model:FlowProp, smiles, data_config, args, random=False):
    lr = args.opt_lr
    num_iter = args.num_iter
    
    if args.property_name == 'qed':
        propf = env.qed  # [0,1]
    elif args.property_name == 'plogp':
        propf = env.penalized_logp  # unbounded, normalized later???
    else:
        raise ValueError("Wrong property_name{}".format(args.property_name))
    property_model.eval()
    with torch.no_grad():
        atoms, bond = smiles_to_adj(smiles, args.dataset)
        atoms, bond = get_mol_data(atoms, bond, data_config)
        atoms, bond = torch.from_numpy(atoms).unsqueeze(0), torch.from_numpy(bond).unsqueeze(0)
        atoms, bond = atoms.to(args.device), bond.to(args.device)
        mol_z, _ = property_model.encode(atoms, bond)
        if args.debug:
            h = property_model.model.to_latent_format(mol_z)
            x_rev, adj_rev = property_model.reverse(h)
            reverse_smiles = adj_to_smiles(x_rev.cpu(), adj_rev.cpu(), num2atom, atom_valency)
            print(smiles, reverse_smiles)
            z, _, _,  = property_model.model(atoms, bond)
            x_rev, adj_rev = property_model.model.reverse(z)
            reverse_smiles2 = adj_to_smiles(x_rev.cpu(), adj_rev.cpu(), num2atom, atom_valency)
            train_smiles2 = adj_to_smiles(atoms.cpu(), bond.cpu(), num2atom, atom_valency)

            print(train_smiles2, reverse_smiles2)
            
    mol = Chem.MolFromSmiles(smiles)
    fp1 = AllChem.GetMorganFingerprint(mol, 2)
    start = (smiles, propf(mol), None) 
    mol_x, mol_adj = mol_z
    
    cur_xs = [x.clone().detach().requires_grad_(True).to(args.device) for x in mol_x]
    cur_adjs = [adj.clone().detach().requires_grad_(True).to(args.device) for adj in mol_adj]
    cur_vec = property_model.model.to_latent_format([cur_xs, cur_adjs])
    
    start_xs = [x.clone().detach().requires_grad_(True).to(args.device) for x in mol_x]
    start_adjs = [adj.clone().detach().requires_grad_(True).to(args.device) for adj in mol_adj]
    start_vec = property_model.model.to_latent_format([start_xs, start_adjs])

    visited = []
    for step in range(num_iter):
        prop_val = property_model.propNN(cur_vec).squeeze()
        grad = torch.autograd.grad(prop_val, cur_vec)[0]
        # cur_vec = cur_vec.data + lr * grad.data
        if random:
            rad = torch.randn_like(cur_vec.data)
            cur_vec = start_vec.data + lr * rad / torch.sqrt(rad * rad)
        else:
            cur_vec = cur_vec.data + lr * grad.data / torch.norm(grad.data, dim=-1)
            lr = lr*args.lr_decay
        cur_vec = cur_vec.clone().detach().requires_grad_(True).to(args.device)  # torch.tensor(cur_vec, requires_grad=True).to(mol_vec)
        visited.append(cur_vec)

    hidden_z = torch.cat(visited, dim=0).to(args.device)
    x, adj = property_model.reverse(hidden_z)
    
    val_res = check_validity(x, adj, num2atom, atom_valency, debug=args.debug)
    valid_mols = val_res['valid_mols']
    valid_smiles = val_res['valid_smiles']
    results = [[], [], [], []]
    sm_set = set()
    sm_set.add(smiles)
    for m, s in zip(valid_mols, valid_smiles):
        if s in sm_set or s == "":
            continue
        sm_set.add(s)
        p = propf(m)
        fp2 = AllChem.GetMorganFingerprint(m, 2)
        sim = DataStructs.TanimotoSimilarity(fp1, fp2)
        if sim >= 0:
            results[0].append((s, p, sim, smiles))
        if sim >= 0.2:
            results[1].append((s, p, sim, smiles))
        if sim >= 0.4:
            results[2].append((s, p, sim, smiles))
        if sim >= 0.6:
            results[3].append((s, p, sim, smiles))
    # smile, property, similarity, mol
    results[0].sort(key=lambda tup: tup[1], reverse=True)
    results[1].sort(key=lambda tup: tup[1], reverse=True)
    results[2].sort(key=lambda tup: tup[1], reverse=True)
    results[3].sort(key=lambda tup: tup[1], reverse=True)
    return results, start

def load_property_csv(data_name, normalize=True):
    """
    We use qed and plogp in zinc250k_property.csv which are recalculated by rdkit
    the recalculated qed results are in tiny inconsistent with qed in zinc250k.csv
    e.g
    zinc250k_property.csv:
    qed,plogp,smile
    0.7319008436872337,3.1399057164163766,CC(C)(C)c1ccc2occ(CC(=O)Nc3ccccc3F)c2c1
    0.9411116113894995,0.17238635659148804,C[C@@H]1CC(Nc2cncc(-c3nncn3C)c2)C[C@@H](C)C1
    import rdkit
    m = rdkit.Chem.MolFromSmiles('CC(C)(C)c1ccc2occ(CC(=O)Nc3ccccc3F)c2c1')
    rdkit.Chem.QED.qed(m): 0.7319008436872337
    from mflow.utils.environment import penalized_logp
    penalized_logp(m):  3.1399057164163766
    However, in oringinal:
    zinc250k.csv
    ,smiles,logP,qed,SAS
    0,CC(C)(C)c1ccc2occ(CC(=O)Nc3ccccc3F)c2c1,5.0506,0.702012232801,2.0840945720726807
    1,C[C@@H]1CC(Nc2cncc(-c3nncn3C)c2)C[C@@H](C)C1,3.1137,0.928975488089,3.4320038192747795

    0.7319008436872337 v.s. 0.702012232801
    and no plogp in zinc250k.csv dataset!
    """
    if data_name == 'qm9':
        # Total: 133885	 Invalid: 0	 bad_plogp: 0 	 bad_qed: 0
        filename = './data_preprocessed/qm9/qm9_property.csv'
    elif data_name == 'zinc250k':
        # Total: 249455	 Invalid: 0	 bad_plogp: 0 	 bad_qed: 0
        filename = './data_preprocessed/zinc250k/zinc250k_property.csv'

    df = pd.read_csv(filename)  # qed, plogp, smile
    if normalize:
        # plogp: # [-62.52, 4.52]
        m = df['plogp'].mean()  # 0.00026
        std = df['plogp'].std() # 2.05
        mn = df['plogp'].min()
        mx = df['plogp'].max()
        # df['plogp'] = 0.5 * (np.tanh(0.01 * ((df['plogp'] - m) / std)) + 1)  # [0.35, 0.51]
        # df['plogp'] = (df['plogp'] - m) / std
        lower = -10 # -5 # -10
        df['plogp'] = df['plogp'].clip(lower=lower, upper=5)
        df['plogp'] = (df['plogp'] - lower) / (mx-lower)
    if normalize:
        tuples = [tuple(x[0:2]) for x in df.values]
    else:
        tuples = [tuple(x) for x in df.values]
    print('Load {} done, length: {}'.format(filename, len(tuples)))
    return tuples


def write_similes(filename, data):
    """
    QM9: Total: 133885	 bad_plogp: 133885 	 bad_qed: 142   plogp is not applicable to the QM9 dataset
    zinc250k:
    :param filename:
    :param data:
    :param atomic_num_list:
    :return:
    """
    f = open(filename, "w")  # append mode
    results = []
    total = 0
    bad_qed = 0
    bad_plogp= 0
    invalid = 0
    for i, r in enumerate(data):
        total += 1
        x, adj, label = r
        mol0 = construct_mol(x, adj, num2atom, atom_valency)
        smile = Chem.MolToSmiles(mol0, isomericSmiles=True)  # 'CC(C)(C)C1=CC=C2OC=C(CC(=O)NC3=CC=CC=C3F)C2=C1'
        mol = Chem.MolFromSmiles(smile)
        if mol == None:
            print(i, smile)
            invalid += 1
            qed = -1
            plogp = -999
            smile2 = 'N/A'
            results.append((qed, plogp, smile, smile2))
            f.write('{},{},{},{}\n'.format(qed, plogp, smile, smile2))
            continue

        smile2 = Chem.MolToSmiles(mol, isomericSmiles=True)  # 'CC(C)(C)c1ccc2occ(CC(=O)Nc3ccccc3F)c2c1'
        try:
            qed = env.qed(mol)
        except ValueError as e:
            bad_qed += 1
            qed = -1
            print(i+1, Chem.MolToSmiles(mol, isomericSmiles=True), ' error in qed')

        try:
            plogp = env.penalized_logp(mol)
        except RuntimeError as e:
            bad_plogp += 1
            plogp = -999
            print(i+1, Chem.MolToSmiles(mol, isomericSmiles=True), ' error in penalized_log')

        results.append((qed, plogp, smile, smile2))
        f.write('{},{},{},{}\n'.format(qed, plogp, smile, smile2))
        f.flush()
    f.close()


    results.sort(key=lambda tup: tup[0], reverse=True)
    fv = filename.split('.')
    f = open(fv[0]+'_sortedByQED.'+fv[1], "w")  # append mode
    for r in results:
        qed, plogp, smile, smile2 = r
        f.write('{},{},{},{}\n'.format(qed, plogp, smile, smile2))
        f.flush()
    f.close()

    results.sort(key=lambda tup: tup[1], reverse=True)
    fv = filename.split('.')
    f = open(fv[0] + '_sortedByPlogp.' + fv[1], "w")  # append mode
    for r in results:
        qed, plogp, smile, smile2 = r
        f.write('{},{},{},{}\n'.format(qed, plogp, smile, smile2))
        f.flush()
    f.close()

    print('Dump done!')
    print('Total: {}\t Invalid: {}\t bad_plogp: {} \t bad_qed: {}\n'.format(total, invalid, bad_plogp, bad_qed))

def find_top_score_smiles(property_model, train_prop, data_config, args):
    start_time = time()
    if args.property_name == 'qed':
        col = 0
    elif args.property_name == 'plogp':
        col = 1
    print('Finding top {} score'.format(args.property_name))
    train_prop_sorted = sorted(train_prop, key=lambda tup: tup[col], reverse=True)  # qed, plogp, smile
    result_list = []
    for i, r in tqdm(enumerate(train_prop_sorted), total=len(train_prop_sorted)):
        print("the number of molecue is {}".format(i))
        if i >= args.topk:
            break
        if i % 50 == 0:
            print('Optimization {}/{}, time: {:.2f} seconds'.format(i, args.topk, time() - start_time))
        qed, plogp, smile = r
        results, ori = optimize_mol(property_model, smile, data_config, args, random=False)
        result_list.extend(results[0])  # results: [(smile2, property, sim, smile1), ...]

    result_list.sort(key=lambda tup: tup[1], reverse=True)

    # check novelty
    train_smile = set()
    for i, r in enumerate(train_prop_sorted):
        qed, plogp, smile = r
        train_smile.add(smile)
        mol = Chem.MolFromSmiles(smile)
        smile2 = Chem.MolToSmiles(mol, isomericSmiles=True)
        train_smile.add(smile2)

    result_list_novel = []
    for i, r in enumerate(result_list):
        smile, score, sim, smile_original = r
        if smile not in train_smile:
            result_list_novel.append(r)

    # dump results
    f = open(args.property_name + '_discovered_sorted_{}_{}_{}_{}_{}.csv'.format(args.split, 'test' if args.is_test_idx else 'train', str(args.lr), str(args.num_iter), str(args.lr_decay)), "w")
    for r in result_list_novel:
        smile, score, sim, smile_original = r
        f.write('{},{},{},{}\n'.format(score, smile, sim, smile_original))
        f.flush()
    f.close()
    print('Dump done!')


def constrain_optimization_smiles(property_model, train_prop, data_config, args):
    start_time = time()
    if args.property_name == 'qed':
        col = 0
    elif args.property_name == 'plogp':
        col = 1

    print('Constrained optimization of {} score'.format(args.property_name))
    train_prop_sorted = sorted(train_prop, key=lambda tup: tup[col]) #, reverse=True)  # qed, plogp, smile
    result_list = [[],[],[],[]]
    nfail = [0, 0, 0, 0]
    for i, r in enumerate(train_prop_sorted):
        print("the number of molecue is {}".format(i))
        if i >= args.topk:
            break
        if i % 50 == 0:
            print('Optimization {}/{}, time: {:.2f} seconds'.format(i, args.topk, time() - start_time))
        qed, plogp, smile = r
        results, ori = optimize_mol(property_model, smile,  data_config, args, random=False)
        for t in range(len(results)):
            if len(results[t]) > 0:
                smile2, property2, sim, _ = results[t][0]
                plogp_delta = property2 - plogp
                if plogp_delta >= 0:
                    result_list[t].append((smile2, property2, sim, smile, qed, plogp, plogp_delta))
                else:
                    nfail[t] += 1
                    print('Failure:{}:{}'.format(i, smile))
            else:
                nfail[t] += 1
                print('Failure:{}:{}'.format(i, smile))
                
    for i in range(len(result_list)):
        df = pd.DataFrame(result_list[i],
                        columns=['smile_new', 'prop_new', 'sim', 'smile_old', 'qed_old', 'plogp_old', 'plogp_delta'])

        print(df.describe())
        df.to_csv(args.property_name+'_constrain_optimization_{}_{}_{}_{}_{}_{}.csv'.format(str(i*0.2), args.split, 'test' if args.is_test_idx else 'train', str(args.opt_lr), str(args.num_iter), str(args.lr_decay)), index=False)
        print("For sim > {}:".format(0.2*i))
        print('nfail:{} in total:{}'.format(nfail[i], args.topk))
        print('success rate: {}'.format((args.topk-nfail[i])*1.0/args.topk))

def initialize_from_checkpoint(model, args, train=True):
    checkpoint = torch.load(args.init_checkpoint, map_location=args.device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print('initialize from %s Done!' % args.init_checkpoint)

def get_mol_property(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return  [qed(mol), penalized_logp(mol)]

if __name__ == '__main__':
    start = time()
    print("Start at Time: {}".format(ctime()))

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, default='./save_optimization')
    parser.add_argument('--dataset', type=str, default='zinc250k', choices=['qm9', 'zinc250k'],
                        help='dataset name')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=23, help='random seed')
    parser.add_argument("--property_model_path", type=str, default=None)
    parser.add_argument('--num_data', type=int,
                        default=None, help='num of data to train')
    parser.add_argument('--split', type=str, default="moflow",
                        help='choose the split type')
    parser.add_argument('--is_test_idx', action='store_true', default=False, 
                        help='whether use test_idx')
    
    parser.add_argument('--num_workers', type=int, default=10,
                        help='num works to generate data.')
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument('--order', type=str, default='bfs',
                        help='order of atom')
    # ******model args******
    parser.add_argument('--deq_type', type=str,
                        default='random', help='dequantization methods.')
    parser.add_argument('--deq_scale', type=float, default=0.6,
                        help='dequantization scale.(only for deq_type random)')
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
    parser.add_argument('--st_type', type=str, default='sigmoid',
                        help='architecture of st net, choice: [exp, sigmoid]')
    parser.add_argument('--inv_rotate', action='store_true',
                        default=False, help='whether rotate node feature')
    # ***bond model***
    parser.add_argument('--b_num_flows', type=int, default=2,
                        help='num of flows in bond model')
    parser.add_argument('--filter_size', type=int, default=512,
                        help='num of filter size in AffineCoupling')
    parser.add_argument('--inv_conv', action='store_true',
                        default=False, help='whether use 1*1 conv')
    parser.add_argument('--squeeze_fold', type=int, default=2,
                        help='squeeze fold')
    
    parser.add_argument('--num_iter', type=int, default=200,
                        help='num iter of optimization')
    parser.add_argument('--learn_prior', action='store_true',
                        default=False, help='learn log-var of gaussian prior.')
    parser.add_argument('--init_checkpoint', type=str, default=None,
                    help='initialize from a checkpoint, if None, do not restore')
    parser.add_argument('--lr', type=float, default=0.001, help='Base learning rate')
    parser.add_argument('--opt_lr', type=float, default=0.001, help='optimization learning rate')
    parser.add_argument('--lr_decay', type=float, default=1,
                        help='Learning rate decay, applied every step of the optimization')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='L2 norm for the parameters')
    parser.add_argument('--hidden', type=str, default="512,16",
                        help='Hidden dimension list for output regression')
    parser.add_argument('--max_epochs', type=int, default=5, help='How many epochs to run in total?')

    parser.add_argument("--delta", type=float, default=0.01)
    parser.add_argument("--img_format", type=str, default='svg')
    parser.add_argument("--property_name", type=str, default='plogp', choices=['qed', 'plogp'])
    parser.add_argument('--additive_transformations', type=strtobool, default=False,
                        help='apply only additive coupling layers')
    parser.add_argument('--temperature', type=float, default=1.0,
                        help='temperature of the gaussian distributions')
    parser.add_argument('--ratio', type=float, default=0, help='optimization task and generation task')

    parser.add_argument('--topk', type=int, default=800, help='Top k smiles as seeds')
    parser.add_argument('--debug', type=strtobool, default='true', help='To run optimization with more information')
    parser.add_argument('--gen_num', type=int, default=10000, help='Number of generated molecules')
    parser.add_argument('--topscore', action='store_true', default=False, help='To find top score')
    parser.add_argument('--consopt', action='store_true', default=False, help='To do constrained optimization')

    args = parser.parse_args()
    # configuration
    if args.dataset == 'polymer':
        # polymer
        num2atom = {0: 6, 1: 7, 2: 8, 3: 9, 4: 14, 5: 15, 6: 16}
        atom_valency = {6: 4, 7: 3, 8: 2, 9: 1, 14: 4, 15: 3, 16: 2}
    else:
        # zinc250k
        num2atom = {0: 6, 1: 7, 2: 8, 3: 9, 4: 15, 5: 16, 6: 17, 7: 35, 8: 53}
        atom_valency = {6: 4, 7: 3, 8: 2, 9: 1, 15: 3, 16: 2, 17: 1, 35: 1, 53: 1}

    args.strides = [2, 2, 2]
    data_path = os.path.join('./data_preprocessed', args.dataset)
    with open(os.path.join(data_path, 'config.txt'), 'r') as f:
        data_config = eval(f.read())

    with open("./dataset/zinc250k/{}_idx.json".format(args.split), "r") as f:
        train_idx, valid_idx = json.load(f)
    dataset = PretrainDataset("./data_preprocessed/{}".format(args.dataset), data_config, args)
    train_dataset = deepcopy(dataset)
    train_dataset._indices = train_idx
    valid_dataset = deepcopy(dataset)
    valid_dataset._indices = valid_idx
    
    if not os.path.exists(os.path.join("./data_preprocessed/{}".format(args.dataset), 'zinc250k_property.csv')):
        smiles_list = dataset.all_smiles
        property_list = []
        print(torch.multiprocessing.cpu_count())
        with Pool(processes=torch.multiprocessing.cpu_count()) as pool:
            iter = pool.imap(get_mol_property, smiles_list)
            for idx, data in tqdm(enumerate(iter), total=len(smiles_list)):
                property_list.append(data)
        mol_property = np.array(property_list)
        table = pd.DataFrame(mol_property, columns=['qed', 'plogp'])
        table['smile'] = smiles_list
        table.to_csv(os.path.join("./data_preprocessed/{}".format(args.dataset), 'zinc250k_property.csv'), index=False)
    
    if args.hidden in ('', ','):
        hidden = []
    else:
        hidden = [int(d) for d in args.hidden.strip(',').split(',')]
    print('Hidden dim for output regression: ', hidden)

    if args.property_model_path is None:
        property_list = load_property_csv(args.dataset, normalize=True)
        mol_property = np.array(property_list) 
        train_dataset.is_mol_property = True
        train_dataset.mol_property = mol_property
        valid_dataset.is_mol_property = True
        valid_dataset.mol_property = mol_property
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size,collate_fn=PretrainDataset.collate_fn, num_workers=args.num_workers, drop_last=True)
        valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size,collate_fn=PretrainDataset.collate_fn, num_workers=args.num_workers, drop_last=True)
        print('Prepare data done! Time {:.2f} seconds'.format(time() - start))
        property_model_path = os.path.join(args.model_dir, '{}_{}_{}_{}.pth'.format(args.property_name, args.split, args.dataset, args.ratio))
        
        model = MolHF(data_config, args).to(args.device)
        initialize_from_checkpoint(model, args)
        property_model = FlowProp(model, hidden).to(args.device)
        property_model = fit_model(property_model, train_loader, valid_loader, args, property_model_path)   
    else:
        print("Loading trained regression model for optimization")
        print('Prepare data done! Time {:.2f} seconds'.format(time() - start))
        prop_list = load_property_csv(args.dataset, normalize=False)
        train_prop = [prop_list[i] for i in train_idx]
        test_prop = [prop_list[i] for i in valid_idx]
        property_model_path = os.path.join(args.model_dir, args.property_model_path)
        print("loading {} regression model from: {}".format(args.property_name, property_model_path))
        model = MolHF(data_config, args).to(args.device)
        initialize_from_checkpoint(model, args)
        property_model = FlowProp(model, hidden).to(args.device)
        property_model.load_state_dict(torch.load(property_model_path, map_location=args.device))
        print('Load model done! Time {:.2f} seconds'.format(time() - start))

        property_model.eval()

        if args.topscore:
            print('Finding top score:')
            find_top_score_smiles(property_model, test_prop if args.is_test_idx else train_prop, data_config, args)

        if args.consopt:
            print('Constrained optimization:')
            constrain_optimization_smiles(property_model, test_prop if args.is_test_idx else train_prop, data_config, args)
            
        print('Total Time {:.2f} seconds'.format(time() - start))

