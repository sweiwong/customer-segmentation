# main.py

import pandas as pd
import os
# Importing modules, including perform_pca for optimization
from src.data_cleaning import load_data, calculate_age, clean_and_impute_data
from src.feature_engineering import (
    create_combined_features, 
    encode_and_select_features, 
    FINAL_MODEL_FEATURES
)
from src.model_training import scale_features, perform_pca, fit_kmedoids
from src.utils import plot_clusters_2d 

# --- Configuration ---
# Setting the path for my raw data and where the final segmented output will be saved
DATA_FILE_PATH = 'data/marketing_campaign.csv'
OUTPUT_FILE_PATH = 'processed_customer_segments.csv'

# Set the optimal number of PCA components for clustering. 
# My final analysis showed that clustering on N=2 components (PC1 and PC2) yielded the highest Silhouette Score.
N_CLUSTERING_COMPONENTS = 2 


def run_segmentation_pipeline():
    """
    Executes the full customer segmentation pipeline: Clean -> Engineer -> Model.
    This script encapsulates the entire process from my capstone project.
    """
    if not os.path.exists(DATA_FILE_PATH):
        print(f"ERROR: Raw data file not found at {DATA_FILE_PATH}. Please ensure it is in the 'data/' folder.")
        return

    print("--- Starting Customer Segmentation Pipeline ---")
    
    # 1. Data Cleaning
    # Loading the initial dataset
    data = load_data(DATA_FILE_PATH)
    if data.empty:
        return
        
    # Calculating key features 'Age' and 'Engaged_Days' before general cleaning
    data = calculate_age(data)
    # Applying the cleaning steps: handling income imputation and removing outliers
    cleaned_df = clean_and_impute_data(data.copy())
    
    # Resetting the index to ensure clean alignment during the next steps
    cleaned_df.reset_index(drop=True, inplace=True)
    
    # 2. Feature Engineering & Selection
    # Creating new features like Total_Expenses and combined product spend (FishMeat/Dessert)
    engineered_df = create_combined_features(cleaned_df.copy())
    # Encoding categorical data and selecting the final feature set for the model
    model_df = encode_and_select_features(engineered_df.copy(), FINAL_MODEL_FEATURES)
    
    # 3. Model Training (Scaling, PCA, and Clustering)
    # Scaling the feature matrix to prepare it for distance-based clustering
    df_scaled = scale_features(model_df.copy())
    
    # APPLYING PCA: Reducing dimensionality to the optimal component count (N=2)
    # Using PCA improves clustering performance by reducing noise.
    df_pca_Nd, _ = perform_pca(df_scaled, n_components=N_CLUSTERING_COMPONENTS) 
    
    # Running the final K-Medoids Clustering model on the PCA components
    cluster_labels = fit_kmedoids(df_pca_Nd, n_clusters=4) # Using the optimal k=4 from my final report
    
    # 4. Final Output, Profiling, and Visualization
    
    # Adding the cluster labels back to the feature-engineered data for easy interpretation and profiling
    final_df = engineered_df.iloc[model_df.index].copy() 
    final_df['Cluster'] = cluster_labels
    
    # Generate 2D PCA for Visualization (This uses the same N=2 components as the model)
    df_pca_2d, _ = perform_pca(df_scaled, n_components=2)
    df_pca_2d['Cluster'] = cluster_labels
    
    # Generate and save the cluster visualization plot
    plot_clusters_2d(df_pca_2d, pc_cols=['PC1', 'PC2'], cluster_col='Cluster')
    
    # Saving the final dataset with customer segment assignments for business use
    final_df.to_csv(OUTPUT_FILE_PATH, index=False)
    
    print(f"\n--- Pipeline Complete ---")
    print(f"Final data with {final_df['Cluster'].nunique()} segments saved to {OUTPUT_FILE_PATH}")
    
    # Displaying mean profiles to quickly validate segment characteristics
    print("\nCluster Profiles (Mean Values):\n", final_df.groupby('Cluster')[['Income', 'Total_Expenses', 'Recency']].mean())

if __name__ == "__main__":
    run_segmentation_pipeline()