# src/data_cleaning.py

import pandas as pd
import numpy as np
from datetime import datetime

# --- Data Loading ---
def load_data(file_path: str) -> pd.DataFrame:
    """
    Loads the raw marketing campaign data from a CSV file.
    """
    try:
        df = pd.read_csv(file_path)
        print(f"Data loaded successfully from {file_path}. Shape: {df.shape}")
        return df
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}. Check your path.")
        return pd.DataFrame()


# --- Initial Cleaning and Feature Preparation ---
def calculate_age(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates the 'Age' of the customer based on 'Year_Birth' and 
    adds an 'Engaged_Days' feature based on Dt_Customer.
    """
    current_year = 2014 # Assumed reference year from project context
    df['Age'] = current_year - df['Year_Birth']
    
    # Calculate days since engagement: Fixed date format issue using dayfirst=True
    df['Dt_Customer'] = pd.to_datetime(df['Dt_Customer'], dayfirst=True)
    
    today = datetime(current_year, 1, 1) 
    df['Engaged_Days'] = (today - df['Dt_Customer']).dt.days
    
    # Drop original columns after feature creation (using direct assignment instead of inplace=True)
    df = df.drop(columns=['Year_Birth', 'Dt_Customer'])
    return df


def clean_and_impute_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Performs data cleaning, handles missing 'Income' values, and removes outliers.
    """
    # 1. Handle Missing Values: Impute 'Income' using the median (using assignment for clean code)
    df['Income'] = df['Income'].fillna(df['Income'].median())
    
    # 2. Outlier Removal (based on common practice and my project's findings):
    df = df[df['Income'] < 200000] # Remove extreme income outliers
    df = df[df['Age'] < 100]       # Remove Age outliers
    
    # 3. Drop remaining irrelevant features (using assignment for clean code)
    df = df.drop(columns=['ID', 'Z_CostContact', 'Z_Revenue'], errors='ignore')
    
    print(f"Data cleaning complete. New shape: {df.shape}")
    return df