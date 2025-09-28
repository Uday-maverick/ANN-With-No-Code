import streamlit as st
import pandas as pd
import plotly.express as px
from database.db_handler import db

def render_leaderboard():
    """Render the competition leaderboard"""
    st.header("🏆 Leaderboard")
    
    # Get leaderboard data
    leaderboard_df = db.get_leaderboard(limit=50)
    
    if leaderboard_df.empty:
        st.info("No submissions yet. Be the first to submit a model!")
        return
    
    # Display top 3 with badges
    if len(leaderboard_df) >= 3:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("🥇 1st Place", 
                     f"{leaderboard_df.iloc[0]['test_accuracy']:.4f}",
                     leaderboard_df.iloc[0]['username'])
        
        with col2:
            st.metric("🥈 2nd Place", 
                     f"{leaderboard_df.iloc[1]['test_accuracy']:.4f}",
                     leaderboard_df.iloc[1]['username'])
        
        with col3:
            st.metric("🥉 3rd Place", 
                     f"{leaderboard_df.iloc[2]['test_accuracy']:.4f}",
                     leaderboard_df.iloc[2]['username'])
    
    st.markdown("---")
    
    # Interactive leaderboard table
    st.subheader("Full Leaderboard")
    
    # Format the dataframe for display
    display_df = leaderboard_df.copy()
    display_df['Rank'] = range(1, len(display_df) + 1)
    display_df['Test Accuracy'] = display_df['test_accuracy'].round(4)
    display_df['Train Accuracy'] = display_df['train_accuracy'].round(4)
    display_df['Val Accuracy'] = display_df['val_accuracy'].round(4)
    display_df['Params'] = display_df['total_params'].apply(lambda x: f"{x:,}")
    display_df['Time (s)'] = display_df['training_time'].round(2)
    
    # Select columns to display
    display_columns = ['Rank', 'username', 'Test Accuracy', 'Val Accuracy', 
                      'Train Accuracy', 'Params', 'Time (s)', 'created_at']
    
    st.dataframe(
        display_df[display_columns].rename(columns={'username': 'User', 'created_at': 'Submitted'}),
        use_container_width=True
    )
    
    # Visualization
    st.markdown("---")
    st.subheader("📈 Performance Trends")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Accuracy distribution
        fig_dist = px.histogram(leaderboard_df, x='test_accuracy', 
                               title='Test Accuracy Distribution',
                               labels={'test_accuracy': 'Test Accuracy'})
        st.plotly_chart(fig_dist, use_container_width=True)
    
    with col2:
        # Accuracy vs Parameters scatter plot
        fig_scatter = px.scatter(leaderboard_df, x='total_params', y='test_accuracy',
                                hover_data=['username'],
                                title='Accuracy vs Model Size',
                                labels={'total_params': 'Parameters', 'test_accuracy': 'Test Accuracy'})
        st.plotly_chart(fig_scatter, use_container_width=True)
