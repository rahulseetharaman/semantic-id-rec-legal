import os
import os.path as osp
import numpy as np
import pandas as pd
import torch

from collections import defaultdict
from data.preprocessing import PreprocessingMixin
from torch_geometric.data import HeteroData, InMemoryDataset
from typing import Callable, List, Optional


class IL_PCSR(InMemoryDataset, PreprocessingMixin):
    """
    IL-PCSR: Indian Legal - Precedent Case Retrieval
    
    Adapted as a recommendation dataset where:
    - Queries are treated as user sessions (case queries)
    - Precedents are treated as items (prior judgments to retrieve)
    - Relevant precedent IDs form the ground truth for evaluation
    """
    
    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        force_reload: bool = False,
        split: Optional[str] = None,
        **kwargs
    ) -> None:
        self.split = split  # IL-PCSR doesn't use splits like Amazon, but accept for compatibility
        super().__init__(root, transform, pre_transform, force_reload)
        self.load(self.processed_paths[0], data_cls=HeteroData)
    
    @property
    def raw_file_names(self) -> List[str]:
        return ['queries.parquet', 'precedents.parquet']
    
    @property
    def processed_file_names(self) -> str:
        return 'data.pt'
    
    def download(self) -> None:
        """Download IL-PCSR dataset from Hugging Face (queries and precedents only)"""
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("Please install datasets: pip install datasets")
        
        os.makedirs(self.raw_dir, exist_ok=True)
        
        # Load queries
        ds_queries = load_dataset('Exploration-Lab/IL-PCSR', name='queries')
        queries_split = ds_queries['train_queries']  # Use training queries
        df_queries = queries_split.to_pandas()
        df_queries.to_parquet(osp.join(self.raw_dir, 'queries.parquet'))
        
        # Load precedents only (no statutes)
        ds_precedents = load_dataset('Exploration-Lab/IL-PCSR', name='precedents')
        precedents_split = ds_precedents['precedent_candidates']
        df_precedents = precedents_split.to_pandas()
        df_precedents.to_parquet(osp.join(self.raw_dir, 'precedents.parquet'))
    
    def process(self, max_seq_len=20) -> None:
        """
        Process IL-PCSR into recommendation format (precedent retrieval only).
        Queries -> Users/Sessions
        Precedents -> Items
        """
        data = HeteroData()
        
        # Load data
        queries_df = pd.read_parquet(osp.join(self.raw_dir, 'queries.parquet'))
        precedents_df = pd.read_parquet(osp.join(self.raw_dir, 'precedents.parquet'))
        
        # Rename fields for consistency
        precedents_df = precedents_df.rename(columns={'case_title': 'title'})
        
        # Create item ID mapping
        item_id_map = {old_id: idx for idx, old_id in enumerate(precedents_df['id'].values)}
        
        # Build sequences from queries
        sequences = {"train": defaultdict(list), "test": defaultdict(list)}
        
        def to_list(val):
            """Convert various input types to a list of strings"""
            if val is None:
                return []
            if isinstance(val, np.ndarray):
                val = val.tolist()
            if isinstance(val, float) and np.isnan(val):
                return []
            if not isinstance(val, list):
                val = [val]
            return [str(v) for v in val if v is not None]
        
        for idx, row in queries_df.iterrows():
            # Get relevant precedent IDs only (filter out statutes)
            relevant_precedent_ids = to_list(row.get('relevant_precedent_ids', []))
            relevant_ids = [iid for iid in relevant_precedent_ids if iid in item_id_map]
            
            if len(relevant_ids) < 2:
                continue  # Skip if too few relevant items
            
            # Map to integer IDs
            relevant_int_ids = [item_id_map[iid] for iid in relevant_ids]
            
            # Split: last item for test, rest for train
            train_items = relevant_int_ids[:-1]
            test_item = relevant_int_ids[-1]
            
            # Train sequence
            sequences["train"]["itemId"].append(train_items)
            sequences["train"]["itemId_fut"].append(test_item)
            sequences["train"]["userId"].append(idx)
            
            # Test sequence (use last max_seq_len items)
            test_seq = relevant_int_ids[-max_seq_len-1:-1]
            test_seq_padded = test_seq + [-1] * (max_seq_len - len(test_seq))
            sequences["test"]["itemId"].append(test_seq_padded)
            sequences["test"]["itemId_fut"].append(test_item)
            sequences["test"]["userId"].append(idx)
        
        # Convert sequences to tensors
        for split in ["train", "test"]:
            data["user", "rated", "item"].history = {
                "train": {
                    "itemId": [torch.tensor(seq, dtype=torch.long) for seq in sequences["train"]["itemId"]],
                    "itemId_fut": torch.tensor(sequences["train"]["itemId_fut"], dtype=torch.long),
                    "userId": torch.tensor(sequences["train"]["userId"], dtype=torch.long)
                },
                "test": {
                    "itemId": torch.stack([torch.tensor(seq, dtype=torch.long) for seq in sequences["test"]["itemId"]]),
                    "itemId_fut": torch.tensor(sequences["test"]["itemId_fut"], dtype=torch.long),
                    "userId": torch.tensor(sequences["test"]["userId"], dtype=torch.long)
                }
            }
        
        # Process item features (encode text descriptions)
        # Combine title + first 3 segments of precedent text
        item_texts = []
        for idx, row in precedents_df.iterrows():
            title = row['title']
            text_content = row['text']
            
            # Handle text field (could be list or string)
            if isinstance(text_content, list):
                text_str = ' '.join(text_content[:3])  # First 3 segments
            else:
                text_str = str(text_content)
            
            combined_text = f"{title}. {text_str}"[:512]  # Limit to 512 chars
            item_texts.append(combined_text)
        
        # Encode text features
        item_embeddings = self._encode_text_feature(item_texts)
        
        # Create train/test split for items
        gen = torch.Generator()
        gen.manual_seed(42)
        is_train = torch.rand(len(precedents_df), generator=gen) > 0.05
        
        data['item'].x = item_embeddings
        data['item'].text = np.array(item_texts)
        data['item'].is_train = is_train
        
        self.save([data], self.processed_paths[0])
