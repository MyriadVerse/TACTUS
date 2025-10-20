import argparse
import os
import random
import numpy as np
import torch
import pandas as pd
import glob
import pickle
import hnswlib
import torch.nn.functional as F
from tqdm import tqdm
from multiprocessing import Pool, Process
from typing import Any, Dict
import fasttext
import fasttext.util

from src.dataset import PretrainTableDataset
from src.train import train, load_checkpoint, inference_on_tables


def run_pretrain(hp):
    seed = hp.run_id
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if "tusSmall" in hp.data:
        path = './data/tusSmall/datalake'
    elif hp.data == "tusLarge":
        path = './data/tusLarge/datalake'
    elif hp.data == "santosSmall":
        path = './data/santosSmall/datalake'
    elif hp.data == "wiki":
        path = './data/wiki/datalake'
    elif hp.data == 'santosLarge':
        path = './data/santosLarge/datalake'
    elif hp.data == 'wdc':
        path = './data/wdc/datalake'
    else:
        raise ValueError(f"Unknown dataset: {hp.data}")

    trainset = PretrainTableDataset.from_hp(path, hp)
    train(trainset, hp)


def extract_vectors(dfs, dataFolder, augment, sample, table_order, run_id, epoch=0):
    model_path = f"./model/{dataFolder}/model_{augment}_{sample}_{table_order}_ep@{epoch}_{run_id}.pt"
    ckpt = torch.load(model_path, map_location=torch.device('cuda'))
    model, trainset = load_checkpoint(ckpt)
    return inference_on_tables(dfs, model, trainset, batch_size=512), model


def get_df(dataFolder):
    dataFiles = glob.glob(os.path.join(dataFolder, "*.csv"))
    dataDFs = {}
    for file in dataFiles:
        df = pd.read_csv(file, lineterminator='\n')
        filename = os.path.basename(file)
        dataDFs[filename] = df
    return dataDFs


def run_TACT(hp):
    dataFolder = hp.data
    dirs = {
        'santosSmall': ['query', 'datalake'],
        'santosLarge': ['query', 'datalake'],
        'tusSmall': ['query', 'datalake'],
        'tusLarge': ['query', 'datalake'],
        'wiki': ['query', 'datalake'],
        'wdc': ['query', 'datalake']
    }
    dataDir = dirs.get(dataFolder, None)
    if dataDir is None:
        raise ValueError(f"Unknown dataset: {dataFolder}")

    for dir in dataDir:
        DATAPATH = './data/' + dataFolder
        DATAFOLDER = os.path.join(DATAPATH, dir)
        dfs = get_df(DATAFOLDER)
        dataEmbeds = []

        cl_features, model = extract_vectors(
            list(dfs.values()), dataFolder, hp.augment_op, hp.sample_meth,
            hp.table_order, hp.run_id, epoch=hp.n_epochs
        )

        for i, file in enumerate(dfs):
            cl_features_file = np.array(cl_features[i])
            with torch.no_grad():
                z = torch.tensor(cl_features_file, dtype=torch.float32).to(model.device)
                table_indices = torch.zeros(z.size(0), dtype=torch.long, device=model.device)
                if model.pool == 'attn':
                    table_embedding = model._attention_pool(z, table_indices)
                elif model.pool == 'multihead_attn':
                    table_embedding = model._multihead_attn_pool(z, table_indices)
                else:
                    raise ValueError("Invalid pooling type.")
                table_embedding = model.table_projector(table_embedding)
            dataEmbeds.append((file, table_embedding.squeeze(0)))

        saveDir = dir

        os.makedirs("./embedding", exist_ok=True)
        os.makedirs(f"./embedding/{dataFolder}", exist_ok=True)
        output_path = f"./embedding/{dataFolder}/cl_{saveDir}_{hp.augment_op}_{hp.sample_meth}_{hp.table_order}_ep@{hp.n_epochs}_{hp.run_id}.pkl"

        if hp.save_model:
            pickle.dump(dataEmbeds, open(output_path, "wb"))


MODEL_PATH = None

def preprocess_value(value: Any) -> str:
    if pd.isna(value):
        return ""
    return str(value).lower().strip().replace('\n', ' ').replace('\r', ' ')


def get_column_embedding(column_values: pd.Series, top_n: int = 100) -> np.ndarray:
    if not hasattr(get_column_embedding, 'ft_model'):
        get_column_embedding.ft_model = fasttext.load_model(MODEL_PATH)
    ft_model = get_column_embedding.ft_model

    processed_values = column_values.apply(preprocess_value)
    value_counts = processed_values.value_counts()
    if len(value_counts) > top_n:
        value_counts = value_counts.head(top_n)

    total_weight = value_counts.sum()
    weighted_embeddings = np.zeros(ft_model.get_dimension())

    for value, count in value_counts.items():
        if not value:
            continue
        emb = ft_model.get_sentence_vector(value)
        weighted_embeddings += emb * count

    return weighted_embeddings / total_weight if total_weight > 0 else weighted_embeddings


def process_single_file(file_path: str, top_n: int):
    filename = os.path.basename(file_path)
    try:
        df = pd.read_csv(file_path, low_memory=False, encoding_errors='ignore')
        col_embeddings = []
        for col in df.columns:
            non_null_vals = df[col].dropna()
            if non_null_vals.empty:
                continue
            col_emb = get_column_embedding(non_null_vals, top_n)
            col_embeddings.append(col_emb)
        return filename, col_embeddings
    except Exception as e:
        print(f"Error processing {filename}: {e}")
        return filename, []


def preprocess_tables(data_dir: str, top_n: int = 100, max_workers: int = 10) -> Dict[str, torch.Tensor]:
    table_embeddings = {}
    file_paths = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.csv')]
    with Pool(processes=max_workers) as pool:
        tasks = [(fp, top_n) for fp in file_paths]
        results = list(tqdm(pool.starmap(process_single_file, tasks), total=len(tasks), desc="Processing tables"))
    for filename, embeddings in results:
        if embeddings:
            tensor_embs = torch.tensor(np.array(embeddings), dtype=torch.float32)
            if tensor_embs.numel() > 0:
                tensor_embs = F.normalize(tensor_embs, p=2, dim=1)
            table_embeddings[filename] = tensor_embs
        else:
            table_embeddings[filename] = torch.tensor([])
    return table_embeddings


def run_value_embedding(hp):
    global MODEL_PATH
    os.environ["TOKENIZERS_PARALLELISM"] = "false"  
    fasttext.util.download_model('en', if_exists='ignore')
    MODEL_PATH = 'cc.en.300.bin'
    _ = fasttext.load_model(MODEL_PATH)

    benchmark_dir = "./data/" + hp.data + "/datalake"
    query_dir = "./data/" + hp.data + "/query"

    benchmark_embeddings = preprocess_tables(benchmark_dir, hp.top_n)
    query_embeddings = preprocess_tables(query_dir, hp.top_n)

    os.makedirs("./embedding", exist_ok=True)
    with open(f"./embedding/{hp.data}/datalake_value_embeddings.pkl", "wb") as f:
        pickle.dump(benchmark_embeddings, f)
    with open(f"./embedding/{hp.data}/query_value_embeddings.pkl", "wb") as f:
        pickle.dump(query_embeddings, f)


def run_index(hp):
    table_path = f"./embedding/{hp.data}/cl_datalake_{hp.augment_op}_{hp.sample_meth}_{hp.table_order}_ep@{hp.n_epochs}_{hp.run_id}.pkl"
    index_path = f"./index/{hp.data}_index.bin"

    with open(table_path, "rb") as f:
        tables = pickle.load(f)

    vec_dim = len(tables[0][1])
    all_vectors = np.array([t[1].detach().cpu().numpy() for t in tables], dtype='float32')

    index = hnswlib.Index(space='cosine', dim=vec_dim)
    index.init_index(max_elements=len(all_vectors), ef_construction=100, M=32)
    index.set_ef(10)
    index.add_items(all_vectors)

    os.makedirs("./index", exist_ok=True)
    index.save_index(index_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="santosSmall")
    parser.add_argument("--logdir", type=str, default="./model")
    parser.add_argument("--run_id", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--pooling", type=str, default="multihead_attn")
    parser.add_argument("--projector_type", type=str, default="mlp")
    parser.add_argument("--max_len", type=int, default=128)
    parser.add_argument("--size", type=int, default=5000)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--temp", type=float, default=0.07)
    parser.add_argument("--n_epochs", type=int, default=3)
    parser.add_argument("--lm", type=str, default='bert')
    parser.add_argument("--projector", type=int, default=768)
    parser.add_argument("--augment_op", type=str, default='sample_table')
    parser.add_argument("--sample_meth", type=str, default='priority_sample')
    parser.add_argument("--save_model", dest="save_model", action="store_true")
    parser.add_argument("--table_order", type=str, default='column')
    parser.add_argument("--fp16", dest="fp16", action="store_true")
    parser.add_argument("--top_n", type=int, default=100)
    hp = parser.parse_args()

    run_pretrain(hp)
    
    p1 = Process(target=run_value_embedding, args=(hp,))
    p2 = Process(target=run_TACT, args=(hp,))
    p1.start()
    p2.start()
    p1.join()
    p2.join()

    run_index(hp)
