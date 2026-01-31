import os
import pickle
import argparse
import torch
from tqdm import tqdm
from typing import Dict, List
from src.search import TACTSearcher
from src.utils import calcMetrics


def run_tact_search(hp, gt_path: str, K: int, k_range: int):
    sampAug = hp.augment_op + "_" + hp.sample_meth
    dataFolder = hp.data
    epoch = hp.n_epochs
    run_id = hp.run_id

    table_path = f"./embedding/{dataFolder}/cl_datalake_{sampAug}_column_ep@{epoch}_{run_id}.pkl"
    query_path = f"./embedding/{dataFolder}/cl_query_{sampAug}_column_ep@{epoch}_{run_id}.pkl"
    index_path = f"./index/{dataFolder}_index.bin"

    searcher = TACTSearcher(table_path, index_path)
    queries = pickle.load(open(query_path, "rb"))

    saveResults = {}

    for q in tqdm(queries):
        res, resLength = searcher.topk(q, K)
        saveResults[q[0]] = res

    os.makedirs("./results", exist_ok=True)
    result_path = f"./results/{hp.data}_{hp.pooling}_results.pkl"
    with open(result_path, 'wb') as f:
        pickle.dump(saveResults, f)

    return result_path


def compute_batch_similarity(query_embs, candidate_embs_list, batch_size=100):
    if query_embs.numel() == 0:
        return [0.0] * len(candidate_embs_list)

    m, d = query_embs.shape
    max_n = 20
    valid_indices, valid_embs = [], []

    for i, emb in enumerate(candidate_embs_list):
        if emb is not None and emb.numel() > 0:
            valid_indices.append(i)
            valid_embs.append(emb)

    valid_scores = []
    num_valid = len(valid_embs)
    for start_idx in range(0, num_valid, batch_size):
        end_idx = min(start_idx + batch_size, num_valid)
        batch_embs = valid_embs[start_idx:end_idx]
        batch_size_curr = len(batch_embs)

        B_batch = torch.zeros(batch_size_curr, max_n, d, dtype=torch.float32)
        mask_batch = torch.zeros(batch_size_curr, max_n, dtype=torch.bool)
        for j, emb in enumerate(batch_embs):
            n_i = min(emb.size(0), max_n)
            B_batch[j, :n_i] = emb[:n_i]
            mask_batch[j, :n_i] = True

        B_transposed = B_batch.permute(0, 2, 1)
        sim_matrix = torch.matmul(query_embs.unsqueeze(0), B_transposed)
        sim_matrix = sim_matrix.masked_fill(~mask_batch.unsqueeze(1), float('-inf'))
        max_sims, _ = torch.max(sim_matrix, dim=2)
        avg_sims = torch.mean(max_sims, dim=1)
        valid_scores.extend(avg_sims.tolist())

    scores = [0.0] * len(candidate_embs_list)
    for idx, score in zip(valid_indices, valid_scores):
        scores[idx] = score
    return scores


def run_rerank(benchmark_path: str,
                query_path: str,
                candidates_path: str,
                gt_path: str,
                K: int,
                k_range: int,
                weight: float) -> Dict[str, List[str]]:
    with open(benchmark_path, "rb") as f:
        normalized_benchmark = pickle.load(f)
    with open(query_path, "rb") as f:
        normalized_query = pickle.load(f)
    with open(candidates_path, 'rb') as f:
        results_dict = pickle.load(f)

    return_result = {}
    for key in tqdm(results_dict.keys()):
        results_values = results_dict[key]
        candidate_embs = [normalized_benchmark.get(table) for _, table in results_values]
        query_embs = normalized_query.get(key, torch.tensor([]))
        sim_scores = compute_batch_similarity(query_embs, candidate_embs)

        candidates = []
        for (origin_score, table), sim in zip(results_values, sim_scores):
            final_score = origin_score + weight * sim
            candidates.append((final_score, table))
        candidates.sort(key=lambda x: x[0], reverse=True)
        return_result[key] = [table for _, table in candidates[:K]]

    if gt_path:
        calcMetrics(K, k_range, return_result, gtPath=gt_path)

    return return_result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default='santosSmall')
    parser.add_argument("--run_id", type=int, default=0)
    parser.add_argument("--n_epochs", type=int, default=3)
    parser.add_argument("--augment_op", type=str, default='sample_table')
    parser.add_argument("--sample_meth", type=str, default='priority_sample')
    parser.add_argument("--pooling", type=str, default='multihead_attn')
    parser.add_argument("--weight", type=float, default=1.0, help="Weight for dual evidence")
    hp = parser.parse_args()

    if hp.data == "santosSmall":
        gt_path = "./data/" + hp.data + "/groundtruth.pkl"
        K, k_range = 10, 2
    elif hp.data in ["wdc", "santosLarge"]:
        gt_path, K, k_range = None, 10, 2
    elif hp.data in ["tusSmall", "tusLarge"]:
        gt_path, K, k_range = "./data/" + hp.data + "/groundtruth.pkl", 60, 10
    elif hp.data == "wiki":
        gt_path, K, k_range = "./data/" + hp.data + "/groundtruth.pkl", 40, 8
    else:
        raise ValueError(f"Unknown dataset: {hp.data}")
    
    candidates_path = run_tact_search(hp, gt_path, K, k_range)

    run_rerank(
        benchmark_path=f"./embedding/{hp.data}/datalake_value_embeddings.pkl",
        query_path=f"./embedding/{hp.data}/query_value_embeddings.pkl",
        candidates_path=candidates_path,
        gt_path=gt_path,
        K=K,
        k_range=k_range,
        weight=hp.weight
    )
