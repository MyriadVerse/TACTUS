import numpy as np
import random
import pickle
import hnswlib
from numpy.linalg import norm
import torch

class TACTSearcher(object):
    def __init__(self,
                 table_path,
                 index_path,
                 scale, 
                 device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        with open(table_path, "rb") as tfile:
            tables = pickle.load(tfile)
        self.tables = random.sample(tables, int(scale*len(tables)))
        print("From %d total data-lake tables, scale down to %d tables" % (len(tables), len(self.tables)))
        self.vec_dim = len(self.tables[1][1])

        all_vectors = np.array([self._tensor_to_numpy(table[1]) for table in self.tables], 
                              dtype='float32')

        self.index = hnswlib.Index(space='cosine', dim=self.vec_dim)
        print("Number of tables in the data-lake: ", len(all_vectors))
        self.index.init_index(max_elements=len(all_vectors), ef_construction=100, M=32)
        self.index.set_ef(10)
        self.index.add_items(all_vectors)
        
        
    
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

    def _to_device(self, tensor):
        if isinstance(tensor, torch.Tensor):
            return tensor.to(self.device)
        return torch.tensor(tensor, device=self.device)
    
    def _tensor_to_numpy(self, tensor):
        if isinstance(tensor, torch.Tensor):
            return tensor.detach().cpu().numpy()
        return np.array(tensor)
