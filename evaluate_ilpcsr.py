"""
Example: Computing MAP, MRR, and F1 metrics for IL-PCSR precedent retrieval

Usage:
    python evaluate_ilpcsr.py --dataset_folder dataset/ilpcsr --model_path out/decoder/ilpcsr/model.pt
"""

import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader

from data.processed import ItemData, SeqData, RecDataset
from data.utils import batch_to
from evaluate.metrics import RetrievalMetrics
from modules.model import EncoderDecoderRetrievalModel
from modules.tokenizer.semids import SemanticIdTokenizer


def evaluate_ilpcsr(dataset_folder, model_path, batch_size=64, top_k=10):
    """
    Evaluate IL-PCSR precedent retrieval with MAP, MRR, F1 metrics.
    
    Args:
        dataset_folder: Path to IL-PCSR dataset
        model_path: Path to trained model checkpoint
        batch_size: Batch size for evaluation
        top_k: Cutoff for evaluation metrics
    """
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load test dataset
    test_dataset = SeqData(
        root=dataset_folder,
        dataset=RecDataset.IL_PCSR,
        is_train=False,
        force_process=False
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize tokenizer first (needed for model initialization)
    tokenizer = SemanticIdTokenizer(
        input_dim=768,
        output_dim=32,
        hidden_dims=[512, 256, 128],
        codebook_size=256,
        n_layers=3,
        n_cat_feats=0,
        rqvae_weights_path="out/rqvae/ilpcsr/checkpoint_49999.pt"
    )
    item_dataset = ItemData(
        root=dataset_folder,
        dataset=RecDataset.IL_PCSR,
        force_process=False
    )
    tokenizer.precompute_corpus_ids(item_dataset)
    tokenizer = tokenizer.to(device)
    
    # Load model
    model = EncoderDecoderRetrievalModel(
        embedding_dim=128,
        attn_dim=384,
        dropout=0.1,
        num_heads=6,
        n_layers=8,
        num_embeddings=256,
        sem_id_dim=tokenizer.sem_ids_dim,
        inference_verifier_fn=lambda x: tokenizer.exists_prefix(x),
        max_pos=test_dataset.max_seq_len * tokenizer.sem_ids_dim,
        jagged_mode=True
    )
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    model = model.to(device)
    model.eval()
    
    # Initialize metrics for precedent retrieval
    metrics = RetrievalMetrics()
    
    # Evaluation loop
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            # Move batch to device
            batch = batch_to(batch, device)
            tokenized_data = tokenizer(batch)
            
            # Generate rankings
            model.enable_generation = True
            generated = model.generate_next_sem_id(tokenized_data, top_k=True, temperature=1)
            
            # Get rankings (convert semantic IDs to item rankings)
            predicted_rankings = generated.sem_ids.cpu().numpy()
            
            # Qrels from ground truth
            actual_items = tokenized_data.sem_ids_fut.cpu().numpy()
            qrels = {idx: set([tuple(item)]) for idx, item in enumerate(actual_items)}
            
            # Accumulate metrics
            metrics.accumulate(
                predicted_rankings=predicted_rankings,
                qrels=qrels,
                top_k=top_k
            )
    
    # Print results
    results = metrics.reduce()
    
    print("\n=== Precedent Retrieval Metrics (IL-PCSR) ===")
    print(f"\nMetrics (Top-{top_k}):")
    print(f"  MAP (Mean Average Precision): {results['map']:.4f}")
    print(f"  MRR (Mean Reciprocal Rank):   {results['mrr']:.4f}")
    print(f"  F1 Score:                     {results['f1']:.4f}")
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_folder", default="dataset/ilpcsr")
    parser.add_argument("--model_path", default="out/decoder/ilpcsr/checkpoint_9999.pt")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--top_k", type=int, default=10)
    
    args = parser.parse_args()
    evaluate_ilpcsr(
        dataset_folder=args.dataset_folder,
        model_path=args.model_path,
        batch_size=args.batch_size,
        top_k=args.top_k
    )
