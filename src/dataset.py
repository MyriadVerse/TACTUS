from argparse import Namespace
import torch
import random
import pandas as pd
import os
from typing import List

from torch.utils import data
from transformers import AutoTokenizer

from .preprocessor import preprocess

lm_mp = {'bert': 'bert-base-uncased'}


class PretrainTableDataset(data.Dataset):
    def __init__(self,
                 path,
                 augment_op,
                 max_len=256,
                 size=None,
                 lm='bert',
                 single_column=False,
                 sample_meth='priority_sample',
                 table_order='column'):
        self.tokenizer = AutoTokenizer.from_pretrained(lm_mp[lm])
        self.max_len = max_len
        self.path = path
        self.augment_op = augment_op
        self.sample_meth = sample_meth
        self.single_column = single_column
        self.table_order = table_order
        self.table_cache = {}

        self.tables = [fn for fn in os.listdir(path) if '.csv' in fn]
        if size is not None:
            self.tables = self.tables[:size]


    @staticmethod
    def from_hp(path: str, hp: Namespace):
        return PretrainTableDataset(path,
                         augment_op=hp.augment_op,
                         lm=hp.lm,
                         max_len=hp.max_len,
                         size=hp.size,
                         single_column=hp.single_column,
                         sample_meth=hp.sample_meth,
                         table_order=hp.table_order)


    def _read_table(self, table_id):
        if table_id in self.table_cache:
            table = self.table_cache[table_id]
        else:
            fn = os.path.join(self.path, self.tables[table_id])
            table = pd.read_csv(fn, lineterminator='\n', low_memory=False)
            self.table_cache[table_id] = table

        return table


    def _tokenize(self, table: pd.DataFrame) -> List[int]:
        res = []
        max_tokens = self.max_len * 2 // len(table.columns)
        budget = max(1, self.max_len // len(table.columns) - 1)
        column_mp = {}

        if self.table_order == 'column':
            for column in table.columns:
                tokens = preprocess(table[column], max_tokens, self.sample_meth)
                col_text = self.tokenizer.cls_token + " " + ' '.join(tokens[:max_tokens]) + " "

                column_mp[column] = len(res)
                res += self.tokenizer.encode(text=col_text, max_length=budget, add_special_tokens=False, truncation=True)
        else:
            raise ValueError(f"Unsupported table order: {self.table_order}")
        
        return res, column_mp


    def __len__(self):
        return len(self.tables)


    def __getitem__(self, idx):
        table_ori = self._read_table(idx)

        if self.single_column:
            col = random.choice(table_ori.columns)
            table_ori = table_ori[[col]]

        table_aug = augment(table_ori, self.augment_op)
        if len(table_ori) > 5:
            sample_size = random.randint(1, len(table_ori)-1)
            table_ori = table_ori.sample(n=sample_size, replace=True)

        x_ori, mp_ori = self._tokenize(table_ori)
        x_aug, mp_aug = self._tokenize(table_aug)

        cls_indices = []
        ori_indices = []
        for col in mp_ori:
            ori_indices.append(mp_ori[col])
        cls_indices.append(ori_indices)
        aug_indices = []
        for col in mp_aug:
            aug_indices.append(mp_aug[col])
        cls_indices.append(aug_indices)

        return x_ori, x_aug, cls_indices


    def pad(self, batch):
        x_ori, x_aug, cls_indices = zip(*batch)
        max_len_ori = max([len(x) for x in x_ori])
        max_len_aug = max([len(x) for x in x_aug])
        maxlen = max(max_len_ori, max_len_aug)
        x_ori_new = [xi + [self.tokenizer.pad_token_id]*(maxlen - len(xi)) for xi in x_ori]
        x_aug_new = [xi + [self.tokenizer.pad_token_id]*(maxlen - len(xi)) for xi in x_aug]

        cls_ori = []
        cls_aug = []
        for item in cls_indices:
            if item != []:
                cls_ori.append(item[0])
                cls_aug.append(item[1])

        return torch.LongTensor(x_ori_new), torch.LongTensor(x_aug_new), (cls_ori, cls_aug)


def augment(table: pd.DataFrame, op: str):
    if op == 'sample_table':
        table = table.copy()
        if len(table) > 5 and len(table.columns) > 2:
            sample_size = random.randint(1, len(table)-1)
            table = table.sample(n=sample_size, replace=True)
            
            num_to_drop = random.randint(1, len(table.columns) - 1)
            cols_to_drop = random.sample(list(table.columns), num_to_drop)
            table = table.drop(columns=cols_to_drop)
    else:
        raise ValueError(f"Unsupported operation: {op}")

    return table