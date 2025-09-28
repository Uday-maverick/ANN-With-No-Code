import tensorflow as tf
import numpy as np
import time
from typing import List, Dict, Any, Tuple
from tensorflow.keras import layers, models, regularizers
from config.settings import MODEL_CONFIG

class ANNModel:
    def __init__(self):
        self.model = None
        self.history = None
        self.metrics = {}
    
    def build_model(self, parameters: Dict[str, Any], input_shape: Tuple[int, ...]) -> None:
        """Build the ANN model based on user parameters"""
        model = models.Sequential()
        
        # Input layer
        model.add(layers.Flatten(input_shape=input_shape))
        
        # Hidden layers
        n_layers = parameters['n_layers']
        neurons_per_layer = parameters['neurons_per_layer']
        activation = parameters['activation']
        dropout_rate = parameters['dropout_rate']
        l2_reg = parameters.get('l2_regularization', 0.0)
        
        for i in range(n_layers):
            # Add Dense layer with optional L2 regularization
            if l2_reg > 0:
                model.add(layers.Dense(
                    neurons_per_layer,
                    activation=activation,
                    kernel_regularizer=regularizers.l2(l2_reg)
                ))
            else:
                model.add(layers.Dense(neurons_per_layer, activation=activation))
            
            # Add Dropout if rate > 0
            if dropout_rate > 0:
                model.add(layers.Dropout(dropout_rate))
        
        # Output layer (fixed for classification)
        output_units = 10  # MNIST has 10 classes
        model.add(layers.Dense(output_units, activation='softmax'))
        
        self.model = model
    
    def compile_model(self, parameters: Dict[str, Any]) -> None:
        """Compile the model with specified optimizer and learning rate"""
        optimizer_name = parameters['optimizer']
        learning_rate = parameters['learning_rate']
        
        optimizers = {
            'sgd': tf.keras.optimizers.SGD(learning_rate=learning_rate),
            'adam': tf.keras.optimizers.Adam(learning_rate=learning_rate),
            'rmsprop': tf.keras.optimizers.RMSprop(learning_rate=learning_rate),
            'adagrad': tf.keras.optimizers.Adagrad(learning_rate=learning_rate)
        }
        
        optimizer = optimizers.get(optimizer_name.lower(), optimizers['adam'])
        
        self.model.compile(
            optimizer=optimizer,
            loss=MODEL_CONFIG.LOSS_FUNCTION,
            metrics=[MODEL_CONFIG.METRIC]
        )
    
    def train_model(self, X_train: np.ndarray, y_train: np.ndarray, 
                   X_val: np.ndarray, y_val: np.ndarray, 
                   parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Train the model and return metrics"""
        epochs = parameters['epochs']
        batch_size = parameters['batch_size']
        
        start_time = time.time()
        
        self.history = self.model.fit(
            X_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(X_val, y_val),
            verbose=0
        )
        
        training_time = time.time() - start_time
        
        # Calculate metrics
        train_loss, train_accuracy = self.model.evaluate(X_train, y_train, verbose=0)
        val_loss, val_accuracy = self.model.evaluate(X_val, y_val, verbose=0)
        
        self.metrics = {
            'train_accuracy': train_accuracy,
            'val_accuracy': val_accuracy,
            'training_time': training_time,
            'total_params': self.model.count_params(),
            'final_loss': self.history.history['loss'][-1],
            'final_val_loss': self.history.history['val_loss'][-1]
        }
        
        return self.metrics
    
    def evaluate_model(self, X_test: np.ndarray, y_test: np.ndarray) -> float:
        """Evaluate model on test set and return accuracy"""
        if self.model is None:
            raise ValueError("Model must be built and trained before evaluation")
        
        test_loss, test_accuracy = self.model.evaluate(X_test, y_test, verbose=0)
        self.metrics['test_accuracy'] = test_accuracy
        return test_accuracy
    
    def get_training_history(self) -> Dict[str, List[float]]:
        """Get training history for plotting"""
        return self.history.history if self.history else {}