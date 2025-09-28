import streamlit as st
import numpy as np
import time
from config.settings import UI_CONFIG
from database.db_handler import db
from models.ann_model import ANNModel
from models.data_loader import data_loader
from utils.validators import ParameterValidator

# Import components
from components.sidebar import render_sidebar
from components.parameter_input import render_parameter_input
from components.leaderboard import render_leaderboard
from components.results import render_training_results, render_user_submissions

# Page configuration
st.set_page_config(
    page_title=UI_CONFIG.PAGE_TITLE,
    page_icon=UI_CONFIG.PAGE_ICON,
    layout=UI_CONFIG.LAYOUT,
    initial_sidebar_state=UI_CONFIG.INITIAL_SIDEBAR_STATE
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

def main():
    """Main application function"""
    
    # Render sidebar
    render_sidebar()
    
    # Check if user is logged in
    if 'username' not in st.session_state:
        st.warning("👆 Please enter your username in the sidebar to continue")
        return
    
    # Main content
    st.markdown('<h1 class="main-header">🧠 No-Code ANN Competition</h1>', unsafe_allow_html=True)
    
    # Navigation tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🏠 Train Model", "🏆 Leaderboard", "📈 Your Results", "ℹ️ About"])
    
    with tab1:
        render_train_tab()
    
    with tab2:
        render_leaderboard()
    
    with tab3:
        user_id = db.get_user_id(st.session_state.username)
        if user_id:
            render_user_submissions(user_id)
        else:
            st.info("Submit a model to see your results here!")
    
    with tab4:
        render_about_tab()

def render_train_tab():
    """Render the model training tab"""
    
    # Parameter input
    parameters = render_parameter_input()
    
    # Validation and training section
    st.markdown("---")
    st.header("🚀 Train Your Model")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Parameter summary
        st.subheader("Configuration Summary")
        st.json(parameters, expanded=False)
    
    with col2:
        # Train button
        train_button = st.button("🎯 Train Model", type="primary", use_container_width=True)
    
    # Validate parameters
    is_valid, message = ParameterValidator.validate_parameters(parameters)
    
    if not is_valid:
        st.error(f"❌ Invalid parameters: {message}")
        return
    
    if train_button:
        # Initialize training
        with st.spinner("Loading dataset..."):
            data = data_loader.get_data()
        
        # Train model
        train_model(parameters, data)

def train_model(parameters: dict, data: dict):
    """Train the model with given parameters and data"""
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Build model
        status_text.text("Building model architecture...")
        model = ANNModel()
        model.build_model(parameters, data['input_shape'])
        model.compile_model(parameters)
        
        progress_bar.progress(30)
        
        # Train model
        status_text.text("Training model...")
        metrics = model.train_model(
            data['X_train'], data['y_train'],
            data['X_val'], data['y_val'],
            parameters
        )
        
        progress_bar.progress(70)
        
        # Evaluate model
        status_text.text("Evaluating on test set...")
        test_accuracy = model.evaluate_model(data['X_test'], data['y_test'])
        
        progress_bar.progress(100)
        status_text.text("Complete!")
        
        # Save submission
        user_id = db.get_user_id(st.session_state.username)
        if not user_id:
            user_id = db.create_user(st.session_state.username, f"{st.session_state.username}@competition.com")
        
        # Combine metrics
        full_metrics = {
            **metrics,
            'test_accuracy': test_accuracy
        }
        
        # Save to database
        success = db.save_submission(user_id, parameters, full_metrics)
        
        if success:
            st.success("🎉 Model trained and submitted successfully!")
            
            # Display results
            history = model.get_training_history()
            render_training_results(full_metrics, history, parameters)
            
            # Show ranking info
            leaderboard_df = db.get_leaderboard(limit=1000)
            user_rank = leaderboard_df[leaderboard_df['username'] == st.session_state.username].index[0] + 1
            total_players = len(leaderboard_df['username'].unique())
            
            st.balloons()
            st.info(f"🏅 You are ranked **#{user_rank}** out of **{total_players}** participants!")
        
        else:
            st.error("❌ Failed to save submission. Please try again.")
    
    except Exception as e:
        st.error(f"❌ Training failed: {str(e)}")
        st.exception(e)

def render_about_tab():
    """Render the about/information tab"""
    
    st.header("ℹ️ About ANN Competition")
    
    st.markdown("""
    ### 🎯 What is this?
    
    A no-code Artificial Neural Network competition platform where participants can:
    - Configure ANN parameters through a simple interface
    - Train models on standardized datasets (MNIST)
    - Compete for the highest test accuracy
    - Learn about neural networks without coding
    
    ### 🏆 Competition Rules
    
    **Fair Play Guidelines:**
    - All models trained on identical dataset splits
    - Maximum 50 epochs per training session
    - Limited model complexity (max 5 layers, 512 neurons/layer)
    - Same hardware resources for all participants
    
    **Evaluation:**
    - Primary metric: Test accuracy on held-out test set
    - Leaderboard updates automatically
    - Multiple submissions allowed
    - Best score counts for ranking
    
    ### 🔧 Technical Details
    
    **Built with:**
    - Frontend: Streamlit
    - Backend: TensorFlow/Keras
    - Database: SQLite
    - Visualization: Plotly
    
    **Fixed Parameters:**
    - Dataset: MNIST (70/15/15 split)
    - Loss: Categorical Crossentropy
    - Metric: Accuracy
    - Output: Softmax (10 classes)
    
    ### 🚀 Getting Started
    
    1. Enter your username in the sidebar
    2. Configure your model parameters in the "Train Model" tab
    3. Click "Train Model" to start training
    4. View your results and leaderboard ranking
    5. Experiment with different configurations to improve!
    """)

if __name__ == "__main__":
    main()