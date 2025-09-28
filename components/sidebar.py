import streamlit as st
from config.settings import UI_CONFIG

def render_sidebar():
    """Render the sidebar with user info and navigation"""
    with st.sidebar:
        st.title("🧠 ANN Competition")
        st.markdown("---")
        
        # User session management
        if 'username' not in st.session_state:
            username = st.text_input("Enter your username:")
            if username:
                st.session_state.username = username.strip()
                st.rerun()
        else:
            st.success(f"Welcome, **{st.session_state.username}**!")
            if st.button("Logout"):
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                st.rerun()
        
        st.markdown("---")
        st.markdown("### 🎯 Competition Rules")
        st.markdown("""
        - Configure your ANN model parameters
        - Train on MNIST dataset
        - Maximize test accuracy
        - Limited to 50 epochs max
        - Fair resource allocation for all
        """)
        
        st.markdown("---")
        st.markdown("### 📊 Dataset Info")
        st.markdown("""
        **MNIST Dataset:**
        - 70,000 handwritten digits
        - 28×28 grayscale images
        - 10 classes (0-9)
        - 70/15/15 train/val/test split
        """)
