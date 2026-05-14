"""
Utility helper functions.

Provides common utilities for logging, configuration, and reproducibility.
"""

import logging
import json
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
import random
import numpy as np


def set_seed(seed: int) -> None:
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def create_logger(
    name: str,
    log_file: Optional[str] = None,
    level: int = logging.INFO,
) -> logging.Logger:
    """
    Create a configured logger.
    
    Args:
        name: Logger name
        log_file: Optional file to save logs
        level: Logging level
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    
    # File handler
    if log_file:
        fh = logging.FileHandler(log_file)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    
    return logger


def save_config(config: Dict[str, Any], path: str) -> None:
    """
    Save configuration to YAML file.
    
    Args:
        config: Configuration dictionary
        path: Path to save configuration
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    logging.info(f"Configuration saved to {path}")


def load_config(path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    with open(path, 'r') as f:
        config = yaml.safe_load(f)
    
    logging.info(f"Configuration loaded from {path}")
    return config


def save_json(data: Dict[str, Any], path: str) -> None:
    """
    Save data to JSON file.
    
    Args:
        data: Data to save
        path: Path to save file
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)


def load_json(path: str) -> Dict[str, Any]:
    """
    Load data from JSON file.
    
    Args:
        path: Path to JSON file
        
    Returns:
        Loaded data
    """
    with open(path, 'r') as f:
        data = json.load(f)
    
    return data
