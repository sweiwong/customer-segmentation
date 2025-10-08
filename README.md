# Capstone Project: Retail Customer Segmentation 🛍️

## Project Overview

This project implements an **Unsupervised Machine Learning (Clustering)** pipeline to segment a retail business's customer base. The goal was to define distinct customer personas (segments) based on their purchasing habits, demographics, and campaign responses. The resulting profiles can be used by the company to inform targeted marketing strategies, resource allocation, and product development.

My solution proposed the adoption of the **K-Medoids Clustering Algorithm** with **4 optimal customer segments**.

---

## 💡 Customer Segments Identified

Based on my analysis and profiling of the final clusters, I identified four distinct customer profiles:

1.  **High-Value Customers:** Customers with the highest overall income and total spending, particularly on premium products like wine. They show low recency, indicating recent purchasing activity.
2.  **High-Potential Customers:** A younger demographic with moderate spending. They have a high number of web visits, suggesting they are influenced by online marketing and are a key segment for future growth.
3.  **High-Loyalty Customers:** Customers characterized by a long engagement period (low Engaged Days) and a stable family status (Together/Married). They are responsive to deals and store purchases.
4.  **Low-Value/Budget Customers:** Customers with the lowest income and lowest total expenses. This group is highly cost-sensitive, often using deals and having high recency (longer time since last purchase).

---

## ⚙️ Technical Approach & Modular Pipeline

The project follows a modular, reproducible data science workflow (ETL $\rightarrow$ Modeling) built with Python scripts. This structure ensures my solution is maintainable and easily reusable.

### Technologies Used

| Category | Tools & Libraries |
| :--- | :--- |
| **Language** | Python 3.x |
| **Core Libraries** | `pandas`, `numpy` |
| **Modeling** | `scikit-learn`, `sklearn-extra` (specifically for K-Medoids) |
| **Dimensionality Reduction** | `PCA` (Principal Component Analysis) |
| **Visualization** | `matplotlib`, `seaborn` (used extensively in notebooks) |

### Code Structure

The entire project is orchestrated via the **`main.py`** script, which imports functions from my modular **`src/`** package:

1.  **`src/data_cleaning.py`**: Handles initial data loading, imputation of missing `Income` values, calculation of `Age` and `Engaged_Days`, and outlier removal (Income $\ge 200\text{K}$ and Age $\ge 100$).
2.  **`src/feature_engineering.py`**: Creates composite features like `Total_Expenses`, `Total_Purchases`, `FishMeat`, and handles one-hot encoding for categorical variables like `Education` and `Marital_Status`.
3.  **`src/model_training.py`**: Applies **`StandardScaler`** to normalize features, and then fits the final **`KMedoids`** model with $k=4$ to generate the cluster labels.
4.  **`main.py`**: Runs the full process, saves the resulting segments, and prints the cluster profiles.

---

## 🚀 How to Run the Project Locally

### 1. Installation

1.  Clone the repository:
    ```bash
    git clone [YOUR_GITHUB_REPO_URL]
    cd Capstone_MIT
    ```
2.  Install dependencies (a Python virtual environment is highly recommended):
    ```bash
    pip install -r requirements.txt
    ```

### 2. Execution

Run the main pipeline script from the root directory:

```bash
python3 main.py