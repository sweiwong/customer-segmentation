# src/model_training.py

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn_extra.cluster import KMedoids
from sklearn.metrics import silhouette_score
import numpy as np

# --- Data Preparation ---
def scale_features(df_model: pd.DataFrame) -> pd.DataFrame:
    """
    Scales the features using StandardScaler.
    """
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df_model)
    df_scaled = pd.DataFrame(scaled_features, columns=df_model.columns)
    
    print("Features successfully scaled using StandardScaler.")
    return df_scaled

def perform_pca(df_scaled: pd.DataFrame, n_components: int = 2):
    """
    Applies Principal Component Analysis (PCA) for dimensionality reduction.
    """
    pca = PCA(n_components=n_components)
    principal_components = pca.fit_transform(df_scaled)
    
    pc_cols = [f'PC{i+1}' for i in range(n_components)]
    df_pca = pd.DataFrame(data=principal_components, columns=pc_cols)
    
    print(f"PCA complete, keeping {n_components} components.")
    return df_pca, pca 

# --- Clustering ---
def fit_kmedoids(df_data: pd.DataFrame, n_clusters: int = 4) -> np.ndarray:
    """
    Fits the K-Medoids clustering model and calculates the silhouette score.
    """
    # Based on your final report, 4 clusters was the proposed solution.
    kmedoids = KMedoids(n_clusters=n_clusters, random_state=42)
    labels = kmedoids.fit_predict(df_data)
    
    score = silhouette_score(df_data, labels)
    print(f"K-Medoids Clustering (k={n_clusters}) finished.")
    print(f"Silhouette Score: {score:.4f}")
    
    return labels
