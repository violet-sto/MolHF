# MolHF: A Hierarchical Normalizing Flow for Molecular Graph Generation 

This is the official implementation for the paper:

[MolHF: A Hierarchical Normalizing Flow for Molecular Graph Generation](https://arxiv.org/abs/2305.08457) (IJCAI 2023)

## Environment:
- Python 3.7
- Pytorch
- torch-geometric
- rdkit
- networkx

## Workflow
### 1. Data Preprocessing
ZINC250K dataset
```
python preprocess.py --dataset zinc250k --in_path ./dataset/zinc250k/zinc250k.smi --out_path ./data_preprocessed/zinc250k
```
Polymer dataset
```
python preprocess.py --dataset polymer --in_path ./dataset/polymer/polymer.smi --out_path ./data_preprocessed/polymer
```

### 2. Training MolHF
#### Training model on the ZINC250K dataset:
```
python main.py \
    --dataset zinc250k --device cuda --deq_scale 0.6 \
    --train --save --batch_size 256 --lr 1e-3 \
    --squeeze_fold 2 --n_block 4 \
    --a_num_flows 6 --num_layers 2 --hid_dim 256 \
    --b_num_flows 3 --filter_size 256 \
    --temperature 0.6 --learn_prior --inv_conv --inv_rotate --condition \
    --gen_num 10000 \
    | tee training_zinc250k_molhf.log
```
#### Or downloading and using our trained models in
```
https://drive.google.com/drive/folders/1bgq0gQIzT4GoEfDj_Z9ZeYmXI873gTDm
```

#### Training model on the Polymer dataset:
```
python main.py \
    --dataset polymer --device cuda --deq_scale 0.6 \
    --train --save --batch_size 256 --lr 1e-3 \
    --squeeze_fold 2 --n_block 6 \
    --a_num_flows 8 --num_layers 4 --hid_dim 128 \
    --b_num_flows 3 --filter_size 128 \
    --temperature 0.6 --learn_prior --inv_conv --inv_rotate \
    --gen_num 10000 \
    | tee training_polymer_molhf.log
```

#### Or downloading and using our trained models in
```
https://drive.google.com/drive/folders/1bgq0gQIzT4GoEfDj_Z9ZeYmXI873gTDm
```

## Citation
If you find this repository useful, please consider citing our work:
```
@inproceedings{zhu2023molhf,
  title={MolHF: A Hierarchical Normalizing Flow for Molecular Graph Generation},
  author={Zhu, Yiheng and Ouyang, Zhenqiu and Liao, Ben and Wu, Jialu and Wu, Yixuan and Hsieh, Chang-Yu and Hou, Tingjun and Wu, Jian},
  booktitle={Proceedings of the Thirty-Second International Joint Conference on Artificial Intelligence},
  pages={5002--5010},
  year={2023},
  month={8},
  url={https://doi.org/10.24963/ijcai.2023/556},
}
```
