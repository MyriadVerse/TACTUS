import hashlib
import heapq
import pandas as pd
import collections

def preprocess(column: pd.Series, max_tokens: int, method: str): 
    tokens = []
    colVals = " ".join(map(str, column.tolist())).split(" ")
    if method == 'priority_simple':
        tokens = cell_priority_sampling(colVals, m=max_tokens, random_seed=42)
    else:
        raise ValueError(f"Unsupported sampling method: {method}")
    return tokens


def cell_priority_sampling(series, m=10, random_seed=42):
    if len(series) == 0:
        return []
    if m > 100:
        m = 100
    freq_counter = collections.Counter(series)

    def hash_function(val, seed=random_seed):
        val_str = str(val).encode('utf-8')
        hash_obj = hashlib.sha1(val_str + str(seed).encode('utf-8'))
        hash_int = int(hash_obj.hexdigest(), 16)
        raw_hash = hash_int / (2 ** 160)
        return 0.8 + 0.2 * raw_hash 
    
    top_m_freqs = heapq.nlargest(m, freq_counter.values())
    if not top_m_freqs:
        return [] 
    freq_min = top_m_freqs[-1]  
    min_possible_priority = freq_min / 1.0  
    priority_values = []
    for val, freq in freq_counter.items():
        if freq < min_possible_priority * 0.8:  
            continue
        h_val = hash_function(val)
        priority = freq / h_val
        priority_values.append((priority, val))

    priority_values.sort(reverse=True) 
    sampled_cells = [val for _, val in priority_values[:m]]
    
    return sampled_cells
    
    