#!/usr/bin/env python
"""
Quick start script for pseudo labeling on MIT300/SALICON datasets.
Generates artificial saliency annotations with quality filtering.
"""

import sys
from pathlib import Path
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.data_loader import MITDataset, SALICONDataset
from src.models import SaliencyModel
from src.pseudo_labeling import (
    generate_pseudo_labels,
    filter_pseudo_labels,
    compute_saliency_statistics
)


def main():
    """Generate and filter pseudo labels for artificial dataset creation."""
    
    # Configuration
    DATASET_TYPE = "mit"  # or "salicon"
    DATASET_PATH = "./data/BenchmarkIMAGES/SM"
    RAW_LABELS_DIR = "./pseudo_labels_raw"
    FILTERED_LABELS_DIR = "./pseudo_labels_filtered"
    CONCENTRATION_THRESHOLD = 0.005  # Adjusted for baseline model
    
    print("=" * 70)
    print("Pseudo Labeling Pipeline with Concentration Filtering")
    print("=" * 70)
    
    # Step 1: Load dataset
    print(f"\n[Step 1] Loading {DATASET_TYPE.upper()} dataset from {DATASET_PATH}...")
    if DATASET_TYPE == "mit":
        dataset = MITDataset(DATASET_PATH)
    else:
        dataset = SALICONDataset(DATASET_PATH)
    
    print(f"✓ Loaded {len(dataset)} images")
    
    # Step 2: Initialize model
    print(f"\n[Step 2] Initializing saliency model for inference...")
    model = SaliencyModel(model_name="baseline", pretrained=True)
    print("✓ Model ready")
    
    # Step 3: Generate raw pseudo labels
    print(f"\n[Step 3] Generating raw saliency predictions...")
    gen_stats = generate_pseudo_labels(
        dataset=dataset,
        model=model,
        output_dir=RAW_LABELS_DIR,
        threshold=0.5,
        use_existing=False
    )
    
    print("\n📊 Generation Statistics:")
    print(f"  Total images: {gen_stats['total']}")
    print(f"  Successfully processed: {gen_stats['processed']}")
    print(f"  Skipped: {gen_stats['skipped']}")
    print(f"  Errors: {gen_stats['errors']}")
    
    # Step 4: Filter by concentration
    print(f"\n[Step 4] Filtering pseudo labels by concentration (threshold={CONCENTRATION_THRESHOLD})...")
    filter_stats = filter_pseudo_labels(
        saliency_maps_dir=RAW_LABELS_DIR,
        output_dir=FILTERED_LABELS_DIR,
        concentration_threshold=CONCENTRATION_THRESHOLD
    )
    
    print("\n🎯 Filtering Results:")
    print(f"  Total predictions: {filter_stats['total']}")
    print(f"  Accepted (high certainty): {filter_stats['accepted']}")
    print(f"  Rejected (low certainty): {filter_stats['rejected']}")
    if filter_stats['total'] > 0:
        acceptance_rate = 100 * filter_stats['accepted'] / filter_stats['total']
    else:
        acceptance_rate = 0
    print(f"  Acceptance rate: {acceptance_rate:.1f}%")
    print(f"  Mean concentration: {filter_stats['mean_concentration']:.4f}")
    
    # Step 5: Compute statistics on filtered labels
    print(f"\n[Step 5] Computing statistics on filtered pseudo labels...")
    saliency_stats = compute_saliency_statistics(FILTERED_LABELS_DIR)
    
    if saliency_stats:
        print("\n📈 Filtered Pseudo Label Statistics:")
        print(f"  Count: {saliency_stats['count']}")
        print(f"  Mean intensity: {saliency_stats['mean']:.4f}")
        print(f"  Std deviation: {saliency_stats['std']:.4f}")
        print(f"  Min value: {saliency_stats['min']:.4f}")
        print(f"  Max value: {saliency_stats['max']:.4f}")
    
    print("\n" + "=" * 70)
    print("✅ Artificial dataset generation complete!")
    print("=" * 70)
    print(f"\nRaw predictions: {RAW_LABELS_DIR}/")
    print(f"Filtered (high-quality) labels: {FILTERED_LABELS_DIR}/")
    print(f"\nUse '{FILTERED_LABELS_DIR}' for training new saliency models.")


if __name__ == "__main__":
    main()
