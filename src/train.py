import torch
import pandas as pd
import os
from tqdm import tqdm

from .model import TACT
from .dataset import PretrainTableDataset

from torch.utils import data
from transformers import AdamW, get_linear_schedule_with_warmup
from typing import List


def train_step(train_iter, model, optimizer, scheduler, scaler, hp):
    for i, batch in tqdm(enumerate(train_iter), total=len(train_iter)):
        x_ori, x_aug, cls_indices = batch
        optimizer.zero_grad()

        if hp.fp16:
            with torch.cuda.amp.autocast():
                loss = model(x_ori, x_aug, cls_indices)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
        else:
            loss = model(x_ori, x_aug, cls_indices)
            loss.backward()
            optimizer.step()

        scheduler.step()


def train(trainset, hp):
    padder = trainset.pad
    train_iter = data.DataLoader(dataset=trainset,
                                 batch_size=hp.batch_size,
                                 shuffle=True,
                                 num_workers=0,
                                 collate_fn=padder)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = TACT(hp, device=device, lm=hp.lm)
    model = model.cuda()
    optimizer = AdamW(model.parameters(), lr=hp.lr)

    if hp.fp16:
        scaler = torch.cuda.amp.GradScaler()
    else:
        scaler = None

    num_steps = (len(trainset) // hp.batch_size) * hp.n_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=num_steps)

    for epoch in tqdm(range(1, hp.n_epochs+1)):
        model.train()
        train_step(train_iter, model, optimizer, scheduler, scaler, hp)

        if hp.save_model:
            directory = os.path.join(hp.logdir, hp.task)
            if not os.path.exists(directory):
                os.makedirs(directory)

            ckpt_path = os.path.join(hp.logdir, hp.task, 'model_'+hp.augment_op+'_'+str(hp.sample_meth)+'_'+str(hp.table_order)+'_ep@'+str(epoch)+'_'+str(hp.run_id)+'.pt')

            ckpt = {'model': model.state_dict(),
                    'hp': hp}
            torch.save(ckpt, ckpt_path)


def inference_on_tables(tables: List[pd.DataFrame],
                        model: TACT,
                        unlabeled: PretrainTableDataset,
                        batch_size=128,
                        total=None):
    total=total if total is not None else len(tables)
    batch = []
    results = []
    for tid, table in tqdm(enumerate(tables), total=total):
        x, _ = unlabeled._tokenize(table.head(1000))

        batch.append((x, x, []))
        if tid == total - 1 or len(batch) == batch_size:
            with torch.no_grad():
                x, _, _ = unlabeled.pad(batch)
                column_vectors = model.inference(x)
                ptr = 0
                for xi in x:
                    current = []
                    for token_id in xi:
                        if token_id == unlabeled.tokenizer.cls_token_id:
                            current.append(column_vectors[ptr].cpu().numpy())
                            ptr += 1
                    results.append(current)

            batch.clear()

    return results


def load_checkpoint(ckpt):
    hp = ckpt['hp']

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = TACT(hp, device=device, lm=hp.lm)
    model = model.to(device)
    model.load_state_dict(ckpt['model'])

    if hp.task == "santos":
        ds_path = 'dir_to_datalake'
    elif hp.task == "santosLarge":
        ds_path = 'dir_to_datalake'
    elif hp.task == "tus":
        ds_path = 'dir_to_datalake'
    elif hp.task == "tusLarge":
        ds_path = 'dir_to_datalake'
    elif hp.task == "wdc":
        ds_path = 'dir_to_datalake'
    elif hp.task == "wiki":
        ds_path = 'dir_to_datalake'
    dataset = PretrainTableDataset.from_hp(ds_path, hp)

    return model, dataset
