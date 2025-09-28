import os
from dataclasses import dataclass
from typing import List, Dict, Any

@dataclass
class ModelConfig:
    """Configuration for model parameters and constraints"""
    # Architecture constraints
    MAX_LAYERS: int = 5
    MIN_LAYERS: int = 1
    MAX_NEURONS: int = 512
    MIN_NEURONS: int = 16
    
    # Training constraints
    MAX_EPOCHS: int = 50
    MIN_EPOCHS: int = 1
    BATCH_SIZES: List[int] = None
    LEARNING_RATES: List[float] = None
    
    # Regularization constraints
    MAX_DROPOUT: float = 0.5
    MIN_DROPOUT: float = 0.0
    
    # Fixed parameters
    DATASET: str = "mnist"
    TRAIN_RATIO: float = 0.7
    VAL_RATIO: float = 0.15
    TEST_RATIO: float = 0.15
    LOSS_FUNCTION: str = "sparse_categorical_crossentropy"
    METRIC: str = "accuracy"
    
    def __post_init__(self):
        if self.BATCH_SIZES is None:
            self.BATCH_SIZES = [16, 32, 64, 128]
        if self.LEARNING_RATES is None:
            self.LEARNING_RATES = [0.0001, 0.001, 0.01, 0.1]

@dataclass
class UIConfig:
    """Configuration for UI elements"""
    PAGE_TITLE: str = "No-Code ANN Competition"
    PAGE_ICON: str = "🧠"
    LAYOUT: str = "wide"
    INITIAL_SIDEBAR_STATE: str = "expanded"

# Global configurations
MODEL_CONFIG = ModelConfig()
UI_CONFIG = UIConfig()