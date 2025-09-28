import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

def render_training_results(metrics: dict, history: dict, parameters: dict):
    """Render training results and metrics"""
    st.header("📊 Training Results")
    
    if not metrics:
        st.warning("No training results to display")
        return
    
    # Key metrics in columns
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Test Accuracy", f"{metrics.get('test_accuracy', 0):.4f}")
    
    with col2:
        st.metric("Validation Accuracy", f"{metrics.get('val_accuracy', 0):.4f}")
    
    with col3:
        st.metric("Training Accuracy", f"{metrics.get('train_accuracy', 0):.4f}")
    
    with col4:
        st.metric("Training Time", f"{metrics.get('training_time', 0):.2f}s")
    
    st.markdown("---")
    
    # Training history plots
    if history:
        fig = make_subplots(rows=1, cols=2, subplot_titles=('Accuracy', 'Loss'))
        
        # Accuracy plot
        if 'accuracy' in history and 'val_accuracy' in history:
            fig.add_trace(
                go.Scatter(y=history['accuracy'], name='Train Accuracy'),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(y=history['val_accuracy'], name='Val Accuracy'),
                row=1, col=1
            )
        
        # Loss plot
        if 'loss' in history and 'val_loss' in history:
            fig.add_trace(
                go.Scatter(y=history['loss'], name='Train Loss'),
                row=1, col=2
            )
            fig.add_trace(
                go.Scatter(y=history['val_loss'], name='Val Loss'),
                row=1, col=2
            )
        
        fig.update_layout(height=400, showlegend=True)
        st.plotly_chart(fig, use_container_width=True)
    
    # Model architecture summary
    st.subheader("🧮 Model Summary")
    
    arch_col1, arch_col2, arch_col3 = st.columns(3)
    
    with arch_col1:
        st.info(f"**Layers:** {parameters['n_layers']} Dense")
        st.info(f"**Neurons/Layer:** {parameters['neurons_per_layer']}")
    
    with arch_col2:
        st.info(f"**Activation:** {parameters['activation']}")
        st.info(f"**Dropout:** {parameters['dropout_rate']}")
    
    with arch_col3:
        st.info(f"**Optimizer:** {parameters['optimizer']}")
        st.info(f"**Learning Rate:** {parameters['learning_rate']}")

def render_user_submissions(user_id: int):
    """Render user's submission history"""
    from database.db_handler import db
    
    submissions_df = db.get_user_submissions(user_id)
    
    if submissions_df.empty:
        st.info("You haven't made any submissions yet.")
        return
    
    st.subheader("📋 Your Submissions")
    
    # Format submissions for display
    display_df = submissions_df.copy()
    display_df['Test Accuracy'] = display_df['test_accuracy'].round(4)
    display_df['Rank'] = display_df['test_accuracy'].rank(ascending=False).astype(int)
    
    st.dataframe(
        display_df[['Rank', 'Test Accuracy', 'training_time', 'total_params', 'created_at']],
        use_container_width=True
    )
    
    # Progress chart
    if len(submissions_df) > 1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=submissions_df['created_at'],
            y=submissions_df['test_accuracy'],
            mode='lines+markers',
            name='Your Progress'
        ))
        fig.update_layout(
            title='Your Accuracy Progress',
            xaxis_title='Submission Time',
            yaxis_title='Test Accuracy'
        )
        st.plotly_chart(fig, use_container_width=True)