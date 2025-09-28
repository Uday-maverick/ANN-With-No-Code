import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from config.settings import MODEL_CONFIG

class DataLoader:
    def __init__(self):
        self.datasets = {
            'mnist': tf.keras.datasets.mnist,
            'fashion_mnist': tf.keras.datasets.fashion_mnist,
            'cifar10': tf.keras.datasets.cifar10
        }
        self.data = None
        self.preprocessed = False
    
    def load_data(self, dataset_name: str = None) -> None:
        """Load and preprocess the dataset"""
        if dataset_name is None:
            dataset_name = MODEL_CONFIG.DATASET
        
        if dataset_name not in self.datasets:
            raise ValueError(f"Dataset {dataset_name} not supported")
        
        # Load dataset
        (X_train, y_train), (X_test, y_test) = self.datasets[dataset_name].load_data()
        
        # Normalize pixel values
        X_train = X_train.astype('float32') / 255.0
        X_test = X_test.astype('float32') / 255.0
        
        # Split training data into train/validation
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train,
            test_size=MODEL_CONFIG.VAL_RATIO,
            random_state=42,
            stratify=y_train
        )
        
        self.data = {
            'X_train': X_train, 'y_train': y_train,
            'X_val': X_val, 'y_val': y_val,
            'X_test': X_test, 'y_test': y_test,
            'input_shape': X_train.shape[1:],
            'num_classes': len(np.unique(y_train))
        }
        
        self.preprocessed = True
    
    def get_data(self) -> dict:
        """Get preprocessed data"""
        if not self.preprocessed:
            self.load_data()
        return self.data

# Global data loader instance
data_loader = DataLoader()