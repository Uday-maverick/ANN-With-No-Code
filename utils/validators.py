from typing import Dict, Any, Tuple
from config.settings import MODEL_CONFIG

class ParameterValidator:
    @staticmethod
    def validate_parameters(parameters: Dict[str, Any]) -> Tuple[bool, str]:
        """Validate user parameters against constraints"""
        try:
            # Validate number of layers
            if not (MODEL_CONFIG.MIN_LAYERS <= parameters['n_layers'] <= MODEL_CONFIG.MAX_LAYERS):
                return False, f"Number of layers must be between {MODEL_CONFIG.MIN_LAYERS} and {MODEL_CONFIG.MAX_LAYERS}"
            
            # Validate neurons per layer
            if not (MODEL_CONFIG.MIN_NEURONS <= parameters['neurons_per_layer'] <= MODEL_CONFIG.MAX_NEURONS):
                return False, f"Neurons per layer must be between {MODEL_CONFIG.MIN_NEURONS} and {MODEL_CONFIG.MAX_NEURONS}"
            
            # Validate dropout rate
            if not (MODEL_CONFIG.MIN_DROPOUT <= parameters['dropout_rate'] <= MODEL_CONFIG.MAX_DROPOUT):
                return False, f"Dropout rate must be between {MODEL_CONFIG.MIN_DROPOUT} and {MODEL_CONFIG.MAX_DROPOUT}"
            
            # Validate epochs
            if not (MODEL_CONFIG.MIN_EPOCHS <= parameters['epochs'] <= MODEL_CONFIG.MAX_EPOCHS):
                return False, f"Epochs must be between {MODEL_CONFIG.MIN_EPOCHS} and {MODEL_CONFIG.MAX_EPOCHS}"
            
            # Validate batch size
            if parameters['batch_size'] not in MODEL_CONFIG.BATCH_SIZES:
                return False, f"Batch size must be one of {MODEL_CONFIG.BATCH_SIZES}"
            
            # Validate learning rate
            if parameters['learning_rate'] not in MODEL_CONFIG.LEARNING_RATES:
                return False, f"Learning rate must be one of {MODEL_CONFIG.LEARNING_RATES}"
            
            return True, "Parameters are valid"
        
        except KeyError as e:
            return False, f"Missing parameter: {e}"
    
    @staticmethod
    def get_default_parameters() -> Dict[str, Any]:
        """Get default parameters for the model"""
        return {
            'n_layers': 2,
            'neurons_per_layer': 128,
            'activation': 'relu',
            'dropout_rate': 0.2,
            'l2_regularization': 0.001,
            'optimizer': 'adam',
            'learning_rate': 0.001,
            'batch_size': 32,
            'epochs': 10
        }