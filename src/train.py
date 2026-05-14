"""
Main training script for toxicity prediction models.

This script provides a unified entry point for training various model architectures.
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, Any
import sys

import numpy as np
import yaml

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from toxicity.data import load_tox21_dataset, featurize_molecules
from toxicity.models import BaselineModel, GNNModel, ModelTrainer
from toxicity.evaluation import compute_roc_auc, compute_accuracy
from toxicity.utils import set_seed, create_logger, load_config, save_config


logger = logging.getLogger(__name__)


def load_data(config: Dict[str, Any]) -> Dict[str, np.ndarray]:
    """
    Load and preprocess data.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Dictionary with train, val, test data
    """
    logger.info("Loading and preprocessing data...")
    
    # Load Tox21 dataset
    features, labels = load_tox21_dataset(
        config["data"]["data_dir"],
        target=config["data"]["target"]
    )
    
    # Normalize features
    from toxicity.data import normalize_features
    features_normalized, norm_params = normalize_features(
        features,
        method=config["data"]["normalization"]["method"]
    )
    
    # Split data
    train_ratio, val_ratio, test_ratio = config["data"]["split_ratio"]
    n_samples = len(features_normalized)
    
    train_idx = int(n_samples * train_ratio)
    val_idx = train_idx + int(n_samples * val_ratio)
    
    train_data = {
        "X": features_normalized[:train_idx],
        "y": labels[:train_idx]
    }
    
    val_data = {
        "X": features_normalized[train_idx:val_idx],
        "y": labels[train_idx:val_idx]
    }
    
    test_data = {
        "X": features_normalized[val_idx:],
        "y": labels[val_idx:]
    }
    
    logger.info(f"Data shapes - Train: {train_data['X'].shape}, "
                f"Val: {val_data['X'].shape}, Test: {test_data['X'].shape}")
    
    return {
        "train": train_data,
        "val": val_data,
        "test": test_data,
    }


def train_model(
    config: Dict[str, Any],
    data: Dict[str, Dict[str, np.ndarray]],
) -> Dict[str, Any]:
    """
    Train the specified model.
    
    Args:
        config: Configuration dictionary
        data: Data dictionary
        
    Returns:
        Training results
    """
    logger.info(f"Training {config['model']['type']} model...")
    
    # Initialize model
    if config["model"]["type"] == "baseline":
        model = BaselineModel(
            model_type=config["model"]["baseline"]["model_type"],
            random_state=config["random_seed"]
        )
    elif config["model"]["type"] == "gnn":
        model = GNNModel(
            architecture=config["model"]["gnn"]["architecture"],
            hidden_dims=config["model"]["gnn"]["hidden_dims"],
            dropout=config["model"]["gnn"]["dropout"],
            learning_rate=config["model"]["gnn"]["learning_rate"],
        )
    else:
        raise ValueError(f"Unknown model type: {config['model']['type']}")
    
    # Initialize trainer
    trainer = ModelTrainer(
        model,
        model_dir=config["output"]["model_dir"],
        results_dir=config["output"]["results_dir"],
        verbose=True,
    )
    
    # Train model
    history = trainer.train(
        data["train"],
        val_data=data["val"],
        epochs=config["training"]["epochs"],
        batch_size=config["training"]["batch_size"],
        patience=config["training"]["early_stopping_patience"],
        save_best=config["training"]["save_best_model"],
    )
    
    # Evaluate on test set
    test_metrics = trainer.evaluate(data["test"])
    
    logger.info(f"Test metrics: {test_metrics}")
    
    # Save results
    results = {
        "config": config,
        "history": history,
        "test_metrics": test_metrics,
    }
    
    trainer.save_results(results, name="training_results")
    
    return results


def main():
    """Main training pipeline."""
    parser = argparse.ArgumentParser(
        description="Train toxicity prediction models"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["baseline", "gnn"],
        help="Model type to train (overrides config)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed (overrides config)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for results (overrides config)"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override config with command-line arguments
    if args.model:
        config["model"]["type"] = args.model
    if args.seed is not None:
        config["random_seed"] = args.seed
    if args.output_dir:
        config["output"]["model_dir"] = f"{args.output_dir}/models"
        config["output"]["results_dir"] = f"{args.output_dir}/results"
    
    # Setup logging
    Path(config["output"]["log_dir"]).mkdir(parents=True, exist_ok=True)
    create_logger(
        "toxicity",
        log_file=config["logging"]["log_file"],
        level=logging.getLevelName(config["logging"]["level"])
    )
    
    logger.info("=" * 80)
    logger.info("Toxicity Prediction Training Pipeline")
    logger.info("=" * 80)
    logger.info(f"Configuration: {args.config}")
    logger.info(f"Model type: {config['model']['type']}")
    logger.info(f"Random seed: {config['random_seed']}")
    
    # Set random seed
    set_seed(config["random_seed"])
    
    # Load data
    data = load_data(config)
    
    # Train model
    results = train_model(config, data)
    
    logger.info("=" * 80)
    logger.info("Training completed successfully!")
    logger.info("=" * 80)
    
    return results


if __name__ == "__main__":
    main()
