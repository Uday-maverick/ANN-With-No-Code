"""
Utility helper functions for the ANN Competition platform.
"""

import json
import time
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
import streamlit as st

def format_accuracy(accuracy: float) -> str:
    """
    Format accuracy as percentage with consistent formatting.
    
    Args:
        accuracy: Accuracy value between 0 and 1
        
    Returns:
        Formatted percentage string
    """
    return f"{accuracy * 100:.2f}%"

def format_time(seconds: float) -> str:
    """
    Format time in seconds to human-readable string.
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted time string
    """
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = seconds // 60
        seconds_remainder = seconds % 60
        return f"{int(minutes)}m {seconds_remainder:.2f}s"
    else:
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        return f"{int(hours)}h {int(minutes)}m"

def format_number(number: int) -> str:
    """
    Format large numbers with K, M suffixes.
    
    Args:
        number: Number to format
        
    Returns:
        Formatted number string
    """
    if number >= 1_000_000:
        return f"{number/1_000_000:.1f}M"
    elif number >= 1_000:
        return f"{number/1_000:.1f}K"
    else:
        return str(number)

def calculate_ranking_stats(leaderboard_df: pd.DataFrame, username: str) -> Dict[str, Any]:
    """
    Calculate ranking statistics for a user.
    
    Args:
        leaderboard_df: Leaderboard dataframe
        username: Username to calculate stats for
        
    Returns:
        Dictionary with ranking statistics
    """
    if leaderboard_df.empty:
        return {}
    
    user_data = leaderboard_df[leaderboard_df['username'] == username]
    total_players = leaderboard_df['username'].nunique()
    
    if user_data.empty:
        return {
            'rank': None,
            'total_players': total_players,
            'top_accuracy': leaderboard_df['test_accuracy'].max() if not leaderboard_df.empty else 0,
            'improvement_needed': 0.0
        }
    
    # Calculate rank (1-based index)
    leaderboard_df = leaderboard_df.sort_values('test_accuracy', ascending=False)
    leaderboard_df = leaderboard_df.drop_duplicates('username', keep='first')
    leaderboard_df['rank'] = range(1, len(leaderboard_df) + 1)
    
    user_rank = leaderboard_df[leaderboard_df['username'] == username]['rank'].iloc[0]
    user_accuracy = user_data['test_accuracy'].iloc[0]
    top_accuracy = leaderboard_df['test_accuracy'].max()
    
    improvement_needed = top_accuracy - user_accuracy if user_accuracy < top_accuracy else 0.0
    
    return {
        'rank': user_rank,
        'total_players': total_players,
        'user_accuracy': user_accuracy,
        'top_accuracy': top_accuracy,
        'improvement_needed': improvement_needed,
        'percentile': ((total_players - user_rank) / total_players) * 100 if total_players > 1 else 100
    }

def generate_model_summary(parameters: Dict[str, Any]) -> str:
    """
    Generate a human-readable model summary from parameters.
    
    Args:
        parameters: Model parameters dictionary
        
    Returns:
        Model summary string
    """
    layers = parameters['n_layers']
    neurons = parameters['neurons_per_layer']
    activation = parameters['activation'].upper()
    optimizer = parameters['optimizer'].upper()
    lr = parameters['learning_rate']
    
    return f"{layers}×{neurons} {activation} | {optimizer} (lr={lr})"

def estimate_training_time(parameters: Dict[str, Any], dataset_size: int = 60000) -> float:
    """
    Estimate training time based on parameters and dataset size.
    
    Args:
        parameters: Model parameters
        dataset_size: Size of training dataset
        
    Returns:
        Estimated training time in seconds
    """
    # Base time per epoch (seconds)
    base_time_per_epoch = 0.5
    
    # Factors affecting training time
    complexity_factor = (parameters['n_layers'] * parameters['neurons_per_layer']) / 1000
    batch_size_factor = 64 / parameters['batch_size']  # Larger batch = faster
    dataset_factor = dataset_size / 60000  # Relative to MNIST
    
    estimated_time = (
        base_time_per_epoch * 
        parameters['epochs'] * 
        complexity_factor * 
        batch_size_factor * 
        dataset_factor
    )
    
    return max(estimated_time, 5)  # Minimum 5 seconds

def validate_email(email: str) -> bool:
    """
    Simple email validation.
    
    Args:
        email: Email address to validate
        
    Returns:
        True if email is valid
    """
    import re
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None

def sanitize_username(username: str) -> str:
    """
    Sanitize username by removing special characters and limiting length.
    
    Args:
        username: Raw username input
        
    Returns:
        Sanitized username
    """
    import re
    # Remove special characters, keep alphanumeric, underscore, hyphen
    sanitized = re.sub(r'[^a-zA-Z0-9_-]', '', username)
    # Limit length
    return sanitized[:20]

def create_progress_bar(value: float, total: float, width: int = 40) -> str:
    """
    Create a text-based progress bar.
    
    Args:
        value: Current value
        total: Total value
        width: Width of progress bar in characters
        
    Returns:
        Progress bar string
    """
    percentage = value / total
    filled = int(width * percentage)
    empty = width - filled
    return f"[{'█' * filled}{'░' * empty}] {percentage:.1%}"

def get_performance_tips(parameters: Dict[str, Any], current_accuracy: float) -> List[str]:
    """
    Generate performance improvement tips based on parameters and current accuracy.
    
    Args:
        parameters: Current model parameters
        current_accuracy: Current test accuracy
        
    Returns:
        List of improvement tips
    """
    tips = []
    
    # Architecture tips
    if parameters['n_layers'] < 3 and current_accuracy < 0.95:
        tips.append("💡 Try increasing the number of layers to 3-4 for better feature learning")
    
    if parameters['neurons_per_layer'] < 64 and current_accuracy < 0.9:
        tips.append("💡 Consider increasing neurons per layer to 64-128 for more capacity")
    
    # Regularization tips
    if parameters['dropout_rate'] == 0 and current_accuracy > 0.85:
        tips.append("💡 Add dropout (0.2-0.3) to reduce overfitting and improve generalization")
    
    if parameters['l2_regularization'] == 0 and current_accuracy > 0.88:
        tips.append("💡 Try L2 regularization (0.001) to prevent overfitting")
    
    # Training tips
    if parameters['learning_rate'] > 0.01 and current_accuracy < 0.8:
        tips.append("💡 Lower learning rate (0.001) for more stable convergence")
    
    if parameters['batch_size'] == 16 and current_accuracy < 0.85:
        tips.append("💡 Increase batch size to 32 or 64 for faster training")
    
    if parameters['epochs'] < 20 and current_accuracy < 0.9:
        tips.append("💡 Train for more epochs (20-30) to allow better convergence")
    
    # Optimizer tips
    if parameters['optimizer'] == 'sgd' and current_accuracy < 0.9:
        tips.append("💡 Switch to Adam optimizer for faster and more stable training")
    
    return tips[:3]  # Return top 3 tips

def calculate_confidence_interval(accuracy: float, n_samples: int, confidence: float = 0.95) -> Tuple[float, float]:
    """
    Calculate confidence interval for accuracy.
    
    Args:
        accuracy: Model accuracy
        n_samples: Number of test samples
        confidence: Confidence level (0.95 for 95%)
        
    Returns:
        Tuple of (lower_bound, upper_bound)
    """
    from scipy import stats
    import math
    
    z_score = stats.norm.ppf(1 - (1 - confidence) / 2)
    std_error = math.sqrt((accuracy * (1 - accuracy)) / n_samples)
    
    margin = z_score * std_error
    lower = max(0, accuracy - margin)
    upper = min(1, accuracy + margin)
    
    return lower, upper

def save_parameters_to_file(parameters: Dict[str, Any], filename: str) -> bool:
    """
    Save model parameters to a JSON file.
    
    Args:
        parameters: Model parameters
        filename: Output filename
        
    Returns:
        True if successful
    """
    try:
        with open(filename, 'w') as f:
            json.dump(parameters, f, indent=2)
        return True
    except Exception as e:
        print(f"Error saving parameters: {e}")
        return False

def load_parameters_from_file(filename: str) -> Optional[Dict[str, Any]]:
    """
    Load model parameters from a JSON file.
    
    Args:
        filename: Input filename
        
    Returns:
        Parameters dictionary or None if error
    """
    try:
        with open(filename, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading parameters: {e}")
        return None

def create_parameter_presets() -> Dict[str, Dict[str, Any]]:
    """
    Create predefined parameter presets for beginners.
    
    Returns:
        Dictionary of parameter presets
    """
    return {
        "beginner": {
            "n_layers": 2,
            "neurons_per_layer": 64,
            "activation": "relu",
            "dropout_rate": 0.0,
            "l2_regularization": 0.0,
            "optimizer": "adam",
            "learning_rate": 0.001,
            "batch_size": 32,
            "epochs": 10
        },
        "balanced": {
            "n_layers": 3,
            "neurons_per_layer": 128,
            "activation": "relu",
            "dropout_rate": 0.2,
            "l2_regularization": 0.001,
            "optimizer": "adam",
            "learning_rate": 0.001,
            "batch_size": 64,
            "epochs": 20
        },
        "advanced": {
            "n_layers": 4,
            "neurons_per_layer": 256,
            "activation": "relu",
            "dropout_rate": 0.3,
            "l2_regularization": 0.01,
            "optimizer": "adam",
            "learning_rate": 0.0005,
            "batch_size": 128,
            "epochs": 30
        }
    }

def check_resource_usage(parameters: Dict[str, Any]) -> Dict[str, Any]:
    """
    Check estimated resource usage for training.
    
    Args:
        parameters: Model parameters
        
    Returns:
        Dictionary with resource estimates
    """
    # Estimate model parameters
    input_size = 784  # MNIST flattened
    output_size = 10
    
    total_params = input_size * parameters['neurons_per_layer']
    for _ in range(parameters['n_layers'] - 1):
        total_params += parameters['neurons_per_layer'] * parameters['neurons_per_layer']
    total_params += parameters['neurons_per_layer'] * output_size
    
    # Estimate memory usage (rough approximation)
    memory_mb = (total_params * 4) / (1024 * 1024)  # 4 bytes per parameter
    
    # Estimate computation
    flops = total_params * parameters['batch_size'] * parameters['epochs']
    
    return {
        'estimated_params': total_params,
        'estimated_memory_mb': memory_mb,
        'estimated_flops': flops,
        'resource_level': 'low' if memory_mb < 10 else 'medium' if memory_mb < 50 else 'high'
    }

def format_leaderboard_position(position: int) -> str:
    """
    Format leaderboard position with appropriate emoji.
    
    Args:
        position: Position in leaderboard (1-based)
        
    Returns:
        Formatted position string
    """
    if position == 1:
        return "🥇 1st"
    elif position == 2:
        return "🥈 2nd"
    elif position == 3:
        return "🥉 3rd"
    else:
        return f"{position}th"

def get_competition_metrics(leaderboard_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculate overall competition metrics.
    
    Args:
        leaderboard_df: Leaderboard dataframe
        
    Returns:
        Competition metrics dictionary
    """
    if leaderboard_df.empty:
        return {}
    
    return {
        'total_submissions': len(leaderboard_df),
        'total_participants': leaderboard_df['username'].nunique(),
        'best_accuracy': leaderboard_df['test_accuracy'].max(),
        'average_accuracy': leaderboard_df['test_accuracy'].mean(),
        'median_accuracy': leaderboard_df['test_accuracy'].median(),
        'most_common_optimizer': leaderboard_df['parameters'].apply(
            lambda x: json.loads(x)['optimizer']
        ).mode().iloc[0] if not leaderboard_df.empty else 'adam'
    }

class Timer:
    """Simple timer context manager for performance measurement."""
    
    def __init__(self, name: str = "Operation"):
        self.name = name
        self.start_time = None
        self.end_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        elapsed = self.end_time - self.start_time
        print(f"{self.name} completed in {elapsed:.2f} seconds")
    
    def elapsed(self) -> float:
        """Get elapsed time in seconds."""
        if self.start_time is None:
            return 0.0
        if self.end_time is None:
            return time.time() - self.start_time
        return self.end_time - self.start_time

def log_submission_attempt(username: str, parameters: Dict[str, Any], success: bool):
    """
    Log submission attempts for debugging and monitoring.
    
    Args:
        username: Username
        parameters: Model parameters
        success: Whether submission was successful
    """
    timestamp = datetime.now().isoformat()
    status = "SUCCESS" if success else "FAILED"
    
    log_entry = {
        'timestamp': timestamp,
        'username': username,
        'status': status,
        'parameters': parameters
    }
    
    # In a production system, this would write to a proper log file/database
    print(f"SUBMISSION_ATTEMPT: {json.dumps(log_entry)}")

# Example usage and testing
if __name__ == "__main__":
    # Test the helper functions
    params = {
        "n_layers": 3,
        "neurons_per_layer": 128,
        "activation": "relu",
        "dropout_rate": 0.2,
        "l2_regularization": 0.001,
        "optimizer": "adam",
        "learning_rate": 0.001,
        "batch_size": 64,
        "epochs": 20
    }
    
    print("Model Summary:", generate_model_summary(params))
    print("Estimated Time:", format_time(estimate_training_time(params)))
    print("Performance Tips:", get_performance_tips(params, 0.85))
    print("Resource Usage:", check_resource_usage(params))