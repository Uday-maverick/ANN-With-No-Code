import streamlit as st
from config.settings import MODEL_CONFIG
from utils.validators import ParameterValidator

def render_parameter_input():
    """Render parameter input form"""
    st.header("🔧 Model Configuration")
    
    # Get default parameters
    default_params = ParameterValidator.get_default_parameters()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Architecture")
        
        n_layers = st.slider(
            "Number of Dense Layers",
            min_value=MODEL_CONFIG.MIN_LAYERS,
            max_value=MODEL_CONFIG.MAX_LAYERS,
            value=default_params['n_layers'],
            help="Number of hidden layers in the network"
        )
        
        neurons_per_layer = st.slider(
            "Neurons per Layer",
            min_value=MODEL_CONFIG.MIN_NEURONS,
            max_value=MODEL_CONFIG.MAX_NEURONS,
            value=default_params['neurons_per_layer'],
            step=16,
            help="Number of neurons in each hidden layer"
        )
        
        activation = st.selectbox(
            "Activation Function",
            options=['relu', 'sigmoid', 'tanh'],
            index=0,
            help="Activation function for hidden layers"
        )
    
    with col2:
        st.subheader("Training")
        
        optimizer = st.selectbox(
            "Optimizer",
            options=['adam', 'sgd', 'rmsprop', 'adagrad'],
            index=0,
            help="Optimization algorithm"
        )
        
        learning_rate = st.selectbox(
            "Learning Rate",
            options=MODEL_CONFIG.LEARNING_RATES,
            index=1,
            help="Learning rate for optimization"
        )
        
        batch_size = st.selectbox(
            "Batch Size",
            options=MODEL_CONFIG.BATCH_SIZES,
            index=1,  # Default to 32
            help="Number of samples per gradient update"
        )
        
        epochs = st.slider(
            "Epochs",
            min_value=MODEL_CONFIG.MIN_EPOCHS,
            max_value=MODEL_CONFIG.MAX_EPOCHS,
            value=default_params['epochs'],
            help="Number of training epochs"
        )
    
    st.subheader("Regularization")
    reg_col1, reg_col2 = st.columns(2)
    
    with reg_col1:
        dropout_rate = st.slider(
            "Dropout Rate",
            min_value=float(MODEL_CONFIG.MIN_DROPOUT),
            max_value=float(MODEL_CONFIG.MAX_DROPOUT),
            value=default_params['dropout_rate'],
            step=0.05,
            help="Fraction of input units to drop"
        )
    
    with reg_col2:
        l2_regularization = st.selectbox(
            "L2 Regularization",
            options=[0.0, 0.001, 0.01, 0.1],
            index=1,
            help="L2 regularization strength"
        )
    
    # Compile parameters
    parameters = {
        'n_layers': n_layers,
        'neurons_per_layer': neurons_per_layer,
        'activation': activation,
        'dropout_rate': dropout_rate,
        'l2_regularization': l2_regularization,
        'optimizer': optimizer,
        'learning_rate': learning_rate,
        'batch_size': batch_size,
        'epochs': epochs
    }
    
    return parameters