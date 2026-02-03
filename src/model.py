import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from torch_scatter import scatter_add

lm_mp = {'bert': 'bert-base-uncased'}


class TACTUS(nn.Module):
    def __init__(self, hp, device='cuda', lm='bert'):
        super().__init__()
        self.hp = hp
        self.bert = AutoModel.from_pretrained(lm_mp[lm])
        self.device = device
        hidden_size = 768
        self.projector = nn.Linear(hidden_size, hp.projector)
        self.bn = nn.BatchNorm1d(hidden_size, affine=False)
        self.fc = nn.Linear(hidden_size * 2, 4)
        
        self.pool = self.hp.pooling
        if self.pool == 'attn':
            self.query = nn.Parameter(torch.randn(hidden_size, 1))
            self.temperature = nn.Parameter(torch.tensor(1.0))
        elif self.pool == 'multihead_attn':
            self.mha = nn.MultiheadAttention(hidden_size, 4)
            self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_size))

        self.projector_type = self.hp.projector_type
        if self.projector_type == "linear":
            self.table_projector = nn.Linear(hidden_size, hp.projector)
        elif self.projector_type == "mlp":
            self.table_projector = nn.Sequential(
                nn.Linear(hidden_size, hp.projector),
                nn.ReLU(), 
                nn.Linear(hp.projector, hp.projector)
            )

        self.criterion = nn.CrossEntropyLoss().to(device)
        self.cls_token_id = AutoTokenizer.from_pretrained(lm_mp[lm]).cls_token_id


    def info_nce_TACTUS_loss(self, features, ori_col_num, n_views, temperature=0.07, ori_table_indices=None, aug_table_indices=None):
        z_ori = features[:ori_col_num]
        z_aug = features[ori_col_num:]
        
        def aggregate(z,table_indices):
            if self.pool == 'attn':
                return self._attention_pool(z, table_indices)
            elif self.pool == 'multihead_attn':
                return self._multihead_attn_pool(z, table_indices)
            else:
                raise ValueError(f"Unsupported pooling method: {self.pool}")
        
        gx_all = self.table_projector(aggregate(z_ori, ori_table_indices))
        hx_all = self.table_projector(aggregate(z_aug, aug_table_indices))
        real_batch_size = gx_all.size(0)
        features = torch.cat([gx_all, hx_all], dim=0)
        features = F.normalize(features, dim=1)
        similarity_matrix = torch.matmul(features, features.T)
        
        labels = torch.cat([torch.arange(real_batch_size) for _ in range(n_views)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float().to(self.device)
        
        potential_positives = (similarity_matrix > 0.9) & (labels == 0)
        safe_neg_mask = ~potential_positives & ~labels.bool()
        mask = torch.eye(similarity_matrix.size(0), dtype=torch.bool, device=self.device)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.size(0), -1)
        labels = labels[~mask].view(similarity_matrix.size(0), -1)
        safe_neg_mask = safe_neg_mask[~mask].view(similarity_matrix.size(0), -1)
        
        positives = similarity_matrix[labels.bool()].view(similarity_matrix.size(0), -1)
        negatives = similarity_matrix.clone()
        negatives[~safe_neg_mask] = -10.0 
        k = max(1, int(0.5 * safe_neg_mask.sum(dim=1).float().mean().item()))
        hard_negatives, _ = negatives.topk(k, dim=1, largest=True)
        new_negatives = torch.full_like(negatives, -10.0)  
        new_negatives[:, :k] = hard_negatives
        
        logits = torch.cat([positives, new_negatives], dim=1) / temperature
        labels = torch.zeros(logits.size(0), dtype=torch.long, device=self.device)
        
        return logits, labels
    

    def _extract_columns(self, x, z, cls_indices=None):
        x_flat = x.view(-1)
        column_vectors = z.view((x_flat.shape[0], -1))

        if cls_indices is None:
            indices = [idx for idx, token_id in enumerate(x_flat) if token_id == self.cls_token_id]
        else:
            indices = []
            seq_len = x.shape[-1]
            for rid in range(len(cls_indices)):
                indices += [idx + rid * seq_len for idx in cls_indices[rid]]

        return column_vectors[indices]


    def inference(self, x):
        x = x.to(self.device)
        z = self.bert(x)[0]
        z = self.projector(z) 
        return self._extract_columns(x, z)


    def forward(self, x_ori, x_aug, cls_indices):
        batch_size = len(x_ori)
        x_ori = x_ori.to(self.device) 
        x_aug = x_aug.to(self.device) 

        x = torch.cat((x_ori, x_aug))
        z = self.bert(x)[0] 

        z_ori = z[:batch_size]
        z_aug = z[batch_size:] 

        cls_ori, cls_aug = cls_indices

        z_ori = self._extract_columns(x_ori, z_ori, cls_ori) 
        z_aug = self._extract_columns(x_aug, z_aug, cls_aug)
        z = torch.cat((z_ori, z_aug))
        z = self.projector(z)

        ori_table_indices = torch.cat([
            torch.full((len(col_indices),), fill_value=table_idx, device=z_ori.device)
            for table_idx, col_indices in enumerate(cls_ori)
        ])
        aug_table_indices = torch.cat([
            torch.full((len(col_indices),), fill_value=table_idx, device=z_aug.device)
            for table_idx, col_indices in enumerate(cls_aug)
        ])
        ori_col = z_ori.shape[0]

        logits, labels = self.info_nce_TACTUS_loss(z, ori_col, 2, ori_table_indices=ori_table_indices, aug_table_indices=aug_table_indices, temperature=self.hp.temp)
        loss = self.criterion(logits, labels)
        return loss


    def _attention_pool(self, z, table_indices):
        attn_scores = torch.matmul(z, self.query).squeeze(-1) / self.temperature
        exp_scores = torch.exp(attn_scores - attn_scores.max())
        sum_exp = scatter_add(exp_scores, table_indices, dim=0)[table_indices]
        weights = exp_scores / (sum_exp + 1e-8) 
        
        weighted_z = z * weights.unsqueeze(1)
        return scatter_add(weighted_z, table_indices, dim=0)


    def _multihead_attn_pool(self, z, table_indices):
        unique_tables = torch.unique(table_indices)
        pooled = torch.zeros(len(unique_tables), z.size(1), device=z.device) 
        
        for i, tbl_idx in enumerate(unique_tables):
            mask = (table_indices == tbl_idx)
            tbl_z = z[mask].unsqueeze(1)  
            
            cls_tokens = self.cls_token.expand(1, 1, -1)
            out, _ = self.mha(
                query=cls_tokens,   
                key=tbl_z,         
                value=tbl_z         
            )
            pooled[i] = out.squeeze(0)
        
        return pooled