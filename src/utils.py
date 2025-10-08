# src/utils.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List

def plot_clusters_2d(df: pd.DataFrame, pc_cols: List[str], cluster_col: str = 'Cluster'):
    """
    Generates a scatter plot of the final clusters based on the first two Principal Components.
    
    Args:
        df: DataFrame containing the PCA components and the cluster labels.
        pc_cols: List of column names for the two principal components (e.g., ['PC1', 'PC2']).
        cluster_col: The name of the column containing the cluster labels.
    """
    plt.figure(figsize=(10, 8))
    
    # Use seaborn to generate a clear scatter plot, color-coded by cluster
    sns.scatterplot(
        x=pc_cols[0], 
        y=pc_cols[1], 
        hue=cluster_col, 
        data=df,
        palette='viridis', # A nice color palette for visibility
        s=100, # Size of the dots
        alpha=0.8
    )
    
    # Set titles and labels
    plt.title('Customer Segments Visualized with PCA', fontsize=16)
    plt.xlabel(f'{pc_cols[0]}', fontsize=12)
    plt.ylabel(f'{pc_cols[1]}', fontsize=12)
    plt.legend(title='Segment', loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Save the figure to the project root for easy viewing
    plt.savefig('cluster_visualization.png')
    plt.show()
    print("Cluster visualization saved as 'cluster_visualization.png'.")


