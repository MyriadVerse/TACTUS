import argparse
import os
import time
import pickle
import pandas as pd
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
import fasttext
import fasttext.util
from typing import Dict, List, Any
import torch
from multiprocessing import Pool



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


def process_single_file(file_path: str, top_n: int) -> tuple:
    filename = os.path.basename(file_path)
    try:
        df = pd.read_csv(file_path, low_memory=False, encoding_errors='ignore')
        
        col_embeddings = []
        for col in df.columns:
            try:
                non_null_vals = df[col].dropna()
                if non_null_vals.empty:
                    continue
                    
                col_emb = get_column_embedding(non_null_vals, top_n)
                col_embeddings.append(col_emb)
            except Exception as e:
                print(f"Error processing column {col} in {filename}: {e}")
                continue
        
        return filename, col_embeddings
    
    except Exception as e:
        print(f"Error processing {filename}: {e}")
        return filename, []

def preprocess_tables(data_dir: str, top_n: int = 100, max_workers: int = None) -> Dict[str, torch.Tensor]:
    table_embeddings = {}
    file_list = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    file_paths = [os.path.join(data_dir, f) for f in file_list]
    
    if max_workers is None:
        max_workers = 10
    
    with Pool(processes=max_workers) as pool:
        tasks = [(fp, top_n) for fp in file_paths]
        results = list(tqdm(
            pool.starmap(process_single_file, tasks),
            total=len(tasks),
            desc="Processing tables"
        ))
    
    for filename, embeddings in results:
        if embeddings:
            tensor_embs = torch.tensor(np.array(embeddings), dtype=torch.float32)
            if tensor_embs.numel() > 0:
                tensor_embs = F.normalize(tensor_embs, p=2, dim=1)
            table_embeddings[filename] = tensor_embs
        else:
            table_embeddings[filename] = torch.tensor([])
    
    return table_embeddings


def offline_processing(benchmark_dir: str, 
                       query_dir: str, 
                       top_n: int = 100,
                       benchmark_output: str = "benchmark_value_embeddings.pkl", 
                       query_output: str = "query_value_embeddings.pkl",
                       max_workers: int = 10) -> None:
    global MODEL_PATH
    
    fasttext.util.download_model('en', if_exists='ignore')
    model_path = 'cc.en.300.bin'
    test_model = fasttext.load_model(model_path)
    print(f"Successfully downloaded and loaded FastText model: {model_path}")
    MODEL_PATH = model_path
    
    benchmark_embeddings = preprocess_tables(benchmark_dir, top_n, max_workers)
    query_embeddings = preprocess_tables(query_dir, top_n, max_workers)
    
    with open(benchmark_output, "wb") as f:
        pickle.dump(benchmark_embeddings, f)
    with open(query_output, "wb") as f:
        pickle.dump(query_embeddings, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--top_n", type=int, default=100, help="Number of top values to consider per column")
    hp = parser.parse_args()


    if hp.task == "santos":
        benchmark_dir = "dir_to_datalake"
        query_dir = "dir_to_query"
    elif hp.task == "wdc":
        benchmark_dir = "dir_to_datalake"
        query_dir = "dir_to_query"
    elif hp.task == "santosLarge":
        benchmark_dir = "dir_to_datalake"
        query_dir = "dir_to_query"
    elif hp.task == "tus":
        benchmark_dir = "dir_to_datalake"
        query_dir = "dir_to_query"
    elif hp.task == "tusLarge":
        benchmark_dir = "dir_to_datalake"
        query_dir = "dir_to_query"
    elif hp.task == "wiki":
        benchmark_dir = "dir_to_datalake"
        query_dir = "dir_to_query"

    offline_processing(
        benchmark_dir, 
        query_dir, 
        top_n=hp.top_n,
        benchmark_output=f"dir_to_embedding/{hp.task}_benchmark_value_embeddings.pkl", 
        query_output=f"dir_to_embedding/{hp.task}_query_value_embeddings.pkl"
    )