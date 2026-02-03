import numpy as np
import pickle
import hnswlib
import torch


class TACTUSSearcher(object):
    def __init__(self,
                 table_path: str,
                 index_path: str):
        self.tables = pickle.load(open(table_path, "rb"))
        self.index = hnswlib.Index(space='cosine', dim=768)
        self.index.load_index(index_path)
    
    
    def topk(self, query, K, similarity_threshold=0.5, drop_threshold=0.2):
        query_vec = self._tensor_to_numpy(query[1]).astype('float32')
        if query_vec.ndim == 1:
            query_vec = query_vec.reshape(1, -1) 
        
        search_k = int(3*K)
        labels, distances = self.index.knn_query(query_vec, k=search_k)
        similarities = [
            (1.0 - float(distances[0][i]), self.tables[labels[0][i]][0])
            for i in range(len(labels[0]))
        ]
        similarities.sort(key=lambda x: x[0], reverse=True)
        
        cutoff = int(K)
        for i in range(1, len(similarities)):
            if (similarities[i][0] < similarity_threshold or 
                (i > 0 and similarities[i-1][0] - similarities[i][0] > drop_threshold)):
                cutoff = i - 1
                break
        cutoff = max(int(K), cutoff)

        return similarities[:cutoff], cutoff


    def _tensor_to_numpy(self, tensor):
        if isinstance(tensor, torch.Tensor):
            return tensor.detach().cpu().numpy()
        return np.array(tensor)
