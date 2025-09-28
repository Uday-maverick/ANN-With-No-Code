import sqlite3
import json
import pandas as pd
from datetime import datetime
from typing import List, Dict, Any, Optional
import streamlit as st

class DatabaseHandler:
    def __init__(self, db_path: str = "competition.db"):
        self.db_path = db_path
        self._init_database()
    
    def _init_database(self):
        """Initialize database tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Users table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                user_id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                email TEXT UNIQUE NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Submissions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS submissions (
                submission_id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                parameters TEXT NOT NULL,
                test_accuracy REAL NOT NULL,
                train_accuracy REAL NOT NULL,
                val_accuracy REAL NOT NULL,
                total_params INTEGER NOT NULL,
                training_time REAL NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (user_id)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def create_user(self, username: str, email: str) -> int:
        """Create a new user and return user_id"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute(
                "INSERT INTO users (username, email) VALUES (?, ?)",
                (username, email)
            )
            user_id = cursor.lastrowid
            conn.commit()
            return user_id
        except sqlite3.IntegrityError:
            # User already exists, return existing user_id
            cursor.execute("SELECT user_id FROM users WHERE username = ?", (username,))
            result = cursor.fetchone()
            return result[0] if result else None
        finally:
            conn.close()
    
    def get_user_id(self, username: str) -> Optional[int]:
        """Get user_id by username"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT user_id FROM users WHERE username = ?", (username,))
        result = cursor.fetchone()
        conn.close()
        
        return result[0] if result else None
    
    def save_submission(self, user_id: int, parameters: Dict, metrics: Dict) -> bool:
        """Save model submission to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                INSERT INTO submissions 
                (user_id, parameters, test_accuracy, train_accuracy, val_accuracy, total_params, training_time)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                user_id,
                json.dumps(parameters),
                metrics['test_accuracy'],
                metrics['train_accuracy'],
                metrics['val_accuracy'],
                metrics['total_params'],
                metrics['training_time']
            ))
            conn.commit()
            return True
        except Exception as e:
            print(f"Error saving submission: {e}")
            return False
        finally:
            conn.close()
    
    def get_leaderboard(self, limit: int = 100) -> pd.DataFrame:
        """Get leaderboard data"""
        conn = sqlite3.connect(self.db_path)
        
        query = '''
            SELECT 
                u.username,
                s.test_accuracy,
                s.train_accuracy,
                s.val_accuracy,
                s.total_params,
                s.training_time,
                s.created_at
            FROM submissions s
            JOIN users u ON s.user_id = u.user_id
            ORDER BY s.test_accuracy DESC
            LIMIT ?
        '''
        
        df = pd.read_sql_query(query, conn, params=(limit,))
        conn.close()
        return df
    
    def get_user_submissions(self, user_id: int) -> pd.DataFrame:
        """Get all submissions for a user"""
        conn = sqlite3.connect(self.db_path)
        
        query = '''
            SELECT 
                submission_id,
                parameters,
                test_accuracy,
                train_accuracy,
                val_accuracy,
                total_params,
                training_time,
                created_at
            FROM submissions
            WHERE user_id = ?
            ORDER BY created_at DESC
        '''
        
        df = pd.read_sql_query(query, conn, params=(user_id,))
        conn.close()
        return df

# Global database instance
db = DatabaseHandler()