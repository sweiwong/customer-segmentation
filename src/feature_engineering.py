# src/feature_engineering.py

import pandas as pd
from typing import List

# Define the list of final features used in your final model for easy reference:
FINAL_MODEL_FEATURES = [
    'Income', 'Age', 'Recency', 'Total_Expenses', 'Total_Purchases',
    'NumWebVisitsMonth', 'FishMeat', 'Dessert',
    'MntWines', 'MntGoldProds', 
    'Is_Parent',
    'Education_Graduation', 'Education_PhD', 
    'Marital_Status_Together', 'Marital_Status_Single' 
    # Add other encoded/selected columns used in your final model here
]


def create_combined_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates new, insightful features by combining existing spend and demographic columns.
    """
    # 1. Total Expenses: The sum of all product spending
    spend_cols = ['MntWines', 'MntFruits', 'MntMeatProducts', 
                  'MntFishProducts', 'MntSweetProducts', 'MntGoldProds']
    df['Total_Expenses'] = df[spend_cols].sum(axis=1)
    
    # 2. Total Purchases: The sum of purchases across different channels
    purchase_cols = ['NumDealsPurchases', 'NumWebPurchases', 
                     'NumCatalogPurchases', 'NumStorePurchases']
    df['Total_Purchases'] = df[purchase_cols].sum(axis=1)
    
    # 3. Combined product categories
    df['FishMeat'] = df['MntFishProducts'] + df['MntMeatProducts']
    df['Dessert'] = df['MntFruits'] + df['MntSweetProducts']
    
    return df

def encode_and_select_features(df: pd.DataFrame, final_features: List[str]) -> pd.DataFrame:
    """
    Encodes categorical features and selects the final set of columns for modeling.
    """
    # Create a feature for whether the customer is a parent
    df['Is_Parent'] = df['Kidhome'] + df['Teenhome']
    
    # One-Hot Encode categorical variables
    df = pd.get_dummies(
        df, 
        columns=['Education', 'Marital_Status'], 
        drop_first=False
    )
    
    # Select the final set of features used for clustering
    model_df = df[final_features]
    
    print(f"Feature engineering and selection complete. Model features: {model_df.shape[1]}")
    return model_df
