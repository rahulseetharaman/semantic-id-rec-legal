from collections import defaultdict
from einops import rearrange
from torch import Tensor
import torch
import numpy as np


class TopKAccumulator:
    def __init__(self, ks=[1, 5, 10]):
        self.ks = ks
        self.reset()

    def reset(self):
        self.total = 0
        self.metrics = defaultdict(int)

    def accumulate(self, actual: Tensor, top_k: Tensor) -> None:
        B, D = actual.shape
        pos_match = (rearrange(actual, "b d -> b 1 d") == top_k)
        for i in range(D):
            match_found, rank = pos_match[...,:i+1].all(axis=-1).max(axis=-1)
            matched_rank = rank[match_found]
            for k in self.ks:
                self.metrics[f"h@{k}_slice_:{i+1}"] += len(matched_rank[matched_rank < k])
            
            match_found, rank = pos_match[...,i:i+1].all(axis=-1).max(axis=-1)
            matched_rank = rank[match_found]
            for k in self.ks:
                self.metrics[f"h@{k}_pos_{i}"] += len(matched_rank[matched_rank < k])
        self.total += B
        
    def reduce(self) -> dict:
        return {k: v/self.total for k, v in self.metrics.items()}


class RetrievalMetrics:
    """Compute MAP, MRR, and F1 for precedent retrieval task"""
    
    def __init__(self):
        """Initialize metrics accumulator for precedent retrieval"""
        self.reset()
    
    def reset(self):
        self.map_scores = []
        self.mrr_scores = []
        self.f1_scores = []
    
    def compute_map(self, predicted_ranking: np.ndarray, relevant_set: set) -> float:
        """
        Compute Mean Average Precision for ranking.
        Args:
            predicted_ranking: (num_predictions, sem_id_dim) ranking of items
            relevant_set: set of relevant semantic ID tuples
        Returns:
            MAP score (0-1)
        """
        if len(relevant_set) == 0:
            return 0.0
        
        ap = 0.0
        num_hits = 0
        
        for rank, pred_seq in enumerate(predicted_ranking, 1):
            pred_tuple = tuple(pred_seq)
            if pred_tuple in relevant_set:
                num_hits += 1
                ap += num_hits / rank
        
        return ap / len(relevant_set) if len(relevant_set) > 0 else 0.0
    
    def compute_mrr(self, predicted_ranking: np.ndarray, relevant_set: set) -> float:
        """
        Compute Mean Reciprocal Rank for ranking.
        Args:
            predicted_ranking: (num_predictions, sem_id_dim) ranking of items
            relevant_set: set of relevant semantic ID tuples
        Returns:
            MRR (0-1)
        """
        for rank, pred_seq in enumerate(predicted_ranking, 1):
            pred_tuple = tuple(pred_seq)
            if pred_tuple in relevant_set:
                return 1.0 / rank
        return 0.0
    
    def compute_f1(self, predicted_ranking: np.ndarray, relevant_set: set, top_k: int = 10) -> float:
        """
        Compute Recall@k for ranking.
        Args:
            predicted_ranking: (num_predictions, sem_id_dim) ranking of items
            relevant_set: set of relevant semantic ID tuples
            top_k: cutoff for evaluation
        Returns:
            Recall@k (0-1)
        """
        if len(relevant_set) == 0:
            return 0.0
        
        for rank, pred_seq in enumerate(predicted_ranking[:top_k], 1):
            pred_tuple = tuple(pred_seq)
            if pred_tuple in relevant_set:
                return 1.0
        return 0.0
    
    def accumulate(
        self,
        predicted_rankings: np.ndarray,
        qrels: dict,
        top_k: int = 10
    ) -> None:
        """
        Accumulate metrics for batch of queries with ranking evaluation.
        
        Args:
            predicted_rankings: (batch_size, num_predictions, sem_id_dim) array of predicted semantic IDs
            qrels: dict mapping query_id -> set of relevant semantic ID tuples
            top_k: cutoff for recall@k evaluation
        """
        if isinstance(predicted_rankings, Tensor):
            predicted_rankings = predicted_rankings.cpu().numpy()
        
        for idx, predicted_ranking in enumerate(predicted_rankings):
            # Get relevant items for this query
            if idx in qrels:
                relevant_set = qrels[idx]
            else:
                relevant_set = set()
            
            # predicted_ranking is now (num_predictions, sem_id_dim)
            map_score = self.compute_map(predicted_ranking, relevant_set)
            mrr_score = self.compute_mrr(predicted_ranking, relevant_set)
            f1_score = self.compute_f1(predicted_ranking, relevant_set, top_k)
            
            self.map_scores.append(map_score)
            self.mrr_scores.append(mrr_score)
            self.f1_scores.append(f1_score)
    
    def reduce(self) -> dict:
        """Return aggregated metrics for precedent retrieval"""
        metrics = {
            'map': np.mean(self.map_scores) if self.map_scores else 0.0,
            'mrr': np.mean(self.mrr_scores) if self.mrr_scores else 0.0,
            'f1': np.mean(self.f1_scores) if self.f1_scores else 0.0,
        }
        return metrics
