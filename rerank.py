import argparse
import pickle
from tqdm import tqdm
from typing import Dict, List, Any
from checkPrecisionRecall import calcMetrics
import torch.nn.functional as F
import torch
from multiprocessing import Pool, cpu_count


def online_processing(benchmark_path: str = "benchmark_value_embeddings.pkl", 
                      query_path: str = "query_value_embeddings.pkl",
                      candidates_path: str = "dir_to_results/results_tus.pkl",
                      gt_path: str = 'dir_to_gt',
                      K: int = 60,
                      k_range: int = 2,
                      weight: float = 1.0,
                      task: str = 'wiki') -> Dict[str, List[str]]:
    with open(benchmark_path, "rb") as f:
        normalized_benchmark = pickle.load(f)
    with open(query_path, "rb") as f:
        normalized_query = pickle.load(f)
    
    with open(candidates_path, 'rb') as f:
        results_dict = pickle.load(f)
    
    def compute_batch_similarity(query_embs, candidate_embs_list, batch_size=100):
        if query_embs.numel() == 0:
            return [0.0] * len(candidate_embs_list)
        
        m, d = query_embs.shape
        max_n = 10
        valid_indices = []
        valid_embs = []
        
        for i, emb in enumerate(candidate_embs_list):
            if emb.numel() > 0: 
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
                n_i = emb.size(0)
                if n_i > max_n:
                    emb = emb[:max_n]
                    n_i = max_n
                B_batch[j, :n_i] = emb
                mask_batch[j, :n_i] = True
            
            B_transposed = B_batch.permute(0, 2, 1) 
            sim_matrix = torch.matmul(
                query_embs.unsqueeze(0),  
                B_transposed             
            )  
            
            sim_matrix = sim_matrix.masked_fill(
                ~mask_batch.unsqueeze(1),  
                float('-inf')
            )
            
            max_sims, _ = torch.max(sim_matrix, dim=2) 
            avg_sims = torch.mean(max_sims, dim=1)     
            valid_scores.extend(avg_sims.tolist())
        
        scores = [0.0] * len(candidate_embs_list)
        for idx, score in zip(valid_indices, valid_scores):
            scores[idx] = score
        
        return scores

    return_result = {}
    
    for idx, key in enumerate(tqdm(results_dict.keys())):
        results_values = results_dict[key]
        candidate_embs = [normalized_benchmark.get(table) for _,table in results_values]
        query_embs = normalized_query.get(key, torch.tensor([]))
        sim_scores = compute_batch_similarity(query_embs, candidate_embs)
        
        candidates = []
        for (origin_score, table), sim in zip(results_values, sim_scores):
            final_score = origin_score + weight * sim
            candidates.append((final_score, table))
        
        candidates.sort(key=lambda x: x[0], reverse=True)
        current_K = min(K, len(candidates))
        return_result[key] = [table for _, table in candidates[:current_K]]
    

    if gt_path:
        calcMetrics(K, k_range, return_result, gtPath=gt_path)

    total_candidates = sum(len(v) for v in results_dict.values())
    print(f"Total candidates processed: {total_candidates}")  
    
    return return_result




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--pooling", type=str, default="multihead_attn")
    parser.add_argument("--weight", type=float, default=1.0, help="Weight for similarity score")
    hp = parser.parse_args()

    if hp.task == "santos":
        gt_path = "dir_to_gt"
        K = 10
        k_range = 2
    elif hp.task == "wdc":
        gt_path = None
        K = 10
        k_range = 2
    elif hp.task == "santosLarge":
        gt_path = None
        K = 10
        k_range = 2
    elif hp.task == "tus":
        gt_path = "dir_to_gt"
        K = 60
        k_range = 10
    elif hp.task == "tusLarge":
        gt_path = "dir_to_gt"
        K = 60
        k_range = 10
    elif hp.task == "wiki":
        gt_path = "dir_to_gt"
        K = 40
        k_range = 8

    
    online_processing(
        benchmark_path=f"dir_to_embedding/{hp.task}_benchmark_value_embeddings.pkl", 
        query_path=f"dir_to_embedding/{hp.task}_query_value_embeddings.pkl",
        candidates_path=f"dir_to_results/{hp.task}_{hp.pooling}_results.pkl",
        gt_path=gt_path,
        K=K,
        k_range=k_range,
        weight=hp.weight,
        task=hp.task
    )