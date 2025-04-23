import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# Path where the SHAP values and columns were saved
shap_reason = "shap_results_multiclass"

# Load column names
column_names = pd.read_csv(f"./{shap_reason}/columns.csv")['Column Names'].values

# Number of classes (from shape.csv)
num_classes = int(np.loadtxt(f"./{shap_reason}/shape.csv"))

# Load SHAP values per class
shap_values_list = []
for i in range(num_classes):
    shap_vals = np.loadtxt(f"./{shap_reason}/shap_class_{i}.csv")
    shap_values_list.append(shap_vals)

# Plotting: Absolute mean SHAP values for each feature and each class
for class_idx, shap_vals in enumerate(shap_values_list):
    mean_abs_shap = np.mean(np.abs(shap_vals), axis=0)

    plt.figure(figsize=(10, 6))
    sorted_idx = np.argsort(mean_abs_shap)[::-1]
    plt.barh(range(len(mean_abs_shap)), mean_abs_shap[sorted_idx], align='center')
    plt.yticks(range(len(mean_abs_shap)), column_names[sorted_idx])
    plt.xlabel('Mean |SHAP Value| (Feature Importance)')
    plt.title(f'SHAP Feature Impact - Class {class_idx} ')
    plt.gca().invert_yaxis()  # Highest at the top
    plt.tight_layout()
    plt.show()
