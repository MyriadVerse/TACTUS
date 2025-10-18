import argparse
import numpy as np
import random
import torch

from src.dataset import PretrainTableDataset
from src.train import train

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="small")
    parser.add_argument("--logdir", type=str, default="results/")
    parser.add_argument("--run_id", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--pooling", type=str, default="multihead_attn")
    parser.add_argument("--projector_type", type=str, default="mlp")
    parser.add_argument("--max_len", type=int, default=128)
    parser.add_argument("--size", type=int, default=5000)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--temp", type=float, default=0.07)
    parser.add_argument("--n_epochs", type=int, default=10)
    parser.add_argument("--lm", type=str, default='bert')
    parser.add_argument("--projector", type=int, default=768)
    parser.add_argument("--augment_op", type=str, default='sample_table')
    parser.add_argument("--sample_meth", type=str, default='priority_sample')
    parser.add_argument("--save_model", dest="save_model", action="store_true")
    parser.add_argument("--fp16", dest="fp16", action="store_true")
    parser.add_argument("--single_column", dest="single_column", action="store_true")
    parser.add_argument("--table_order", type=str, default='column')
    
    hp = parser.parse_args()

    seed = hp.run_id
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if "tus" in hp.task:
        path = 'dir_to_datalake'
    elif hp.task == "tusLarge":
        path = 'dir_to_datalake'
    elif hp.task == "santos":
        path = 'dir_to_datalake'
    elif hp.task == "wiki":
        path = 'dir_to_datalake'
    elif hp.task == 'santosLarge':
        path = 'dir_to_datalake'
    elif hp.task == 'wdc':
        path = 'dir_to_datalake'
        
    trainset = PretrainTableDataset.from_hp(path, hp)
    train(trainset, hp)
