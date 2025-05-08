import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
import os


variable_list_df = pd.read_excel('NHIS variable list REVISED 6-14-24_Now_Moddified new.xlsx')
variable_list_df['category'] = variable_list_df['category'].apply(lambda x: x.title() if isinstance(x, str) else x)
column_desc = dict(zip(variable_list_df['variable(s)'].str.upper(),variable_list_df['description']))

import re

custom_labels = {
    'REGION__Midwest': 'Midwest Region of U.S.',
    'REGION__Northeast': 'Northeast Region of U.S.',
    'REGION__South': 'Southern Region of U.S.',
    'REGION__West': 'Western Region of U.S.',
    'ORIENT_A__Bisexual': 'Bisexual Orientation',
    'ORIENT_A__GayLesbian': 'Gay or Lesbian Orientation',
    'ORIENT_A__Straight': 'Straight Orientation',
    'ORIENT_A__Unknown': 'Unknown Sexual Orientation',
    'MARITAL_A__9': 'MARITAL_A__9',
    'MARITAL_A__Married': 'Married',
    'MARITAL_A__Neither': 'Not Married or Living with a partner',
    'MARITAL_A__Unknown': 'Unknown marital status',
    'MARITAL_A__Unmarried couple': 'Living with a partner',
    'RACEALLP_A__AIAN': 'AIAN only',
    'RACEALLP_A__AIAN and any other group': 'AIAN and any other group',
    'RACEALLP_A__Asian': 'Asian only',
    'RACEALLP_A__Black/African-American': 'Black/African American only',
    'RACEALLP_A__Other single and multiple races': 'Other single and multiple races',
    'RACEALLP_A__Unknown': 'Unknown race',
    'RACEALLP_A__White': 'White only'
}

def create_label(inx):
    if inx in custom_labels:
        return custom_labels[inx]
    elif inx.startswith('change_'):
        base = inx.replace('change_', '')
        parts = base.split('__')
        if len(parts) == 2:
            return f"Change in: {column_desc.get(parts[0], parts[0])} [{parts[1].replace('_', ' ')}]"
        else:
            return f"Change in: {column_desc.get(parts[0], parts[0])}"
    else:
        parts = inx.split('__')
        if len(parts) == 2:
            return f"{column_desc.get(parts[0], parts[0])} ({parts[1]})"
        else:
            return column_desc.get(parts[0], parts[0])



# Create output folder if it doesn't exist
os.makedirs("Fig", exist_ok=True)

# Load dataset
print("\ncleaned_data")
Merged_data = pd.read_csv('Data_longtitude_ageEdu.csv')
print('Merged_data: ', Merged_data.shape)

# Extract only 'change_' variables and drop all-zero columns
change_vars = [col for col in Merged_data.columns if col.startswith('change_') and '__NoChange' not in col]

change_data = Merged_data[change_vars]
non_zero_change_vars = change_data.loc[:, (change_data != 0).any()]

# Separate positive and negative changes
positive_changes = non_zero_change_vars.where(non_zero_change_vars > 0)
negative_changes = non_zero_change_vars.where(non_zero_change_vars < 0)

# Normalize (standardize)
scaler = StandardScaler()
positive_scaled = pd.DataFrame(scaler.fit_transform(positive_changes),
                               columns=positive_changes.columns, index=positive_changes.index)
negative_scaled = pd.DataFrame(scaler.fit_transform(negative_changes),
                               columns=negative_changes.columns, index=negative_changes.index)

# Restore NaNs
positive_scaled[positive_changes.isna()] = pd.NA
negative_scaled[negative_changes.isna()] = pd.NA

# Determine global x-axis limits
global_min = min(positive_scaled.min().min(), negative_scaled.min().min())
global_max = max(positive_scaled.max().max(), negative_scaled.max().max())

# Helper: Chunk list into sublists of length n
def chunk_list(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

# ---------- POSITIVE ----------
pos_counts = positive_scaled.notna().sum()
sorted_pos_cols = pos_counts.sort_values(ascending=False).index.tolist()

for i, chunk in enumerate(chunk_list(sorted_pos_cols, 50), start=1):
    chunk_df = positive_scaled[chunk[::-1]]  # Reverse to put higher counts on top

    plt.figure(figsize=(10, max(6, len(chunk) * 0.3)))
    ax = chunk_df.boxplot(vert=False, patch_artist=True, showfliers=False,
                          boxprops=dict(facecolor='lightgreen'))

    # Add count labels
    yticks = ax.get_yticks()
    # Get current tick labels
    yticklabels = [tick.get_text() for tick in ax.get_yticklabels()]
    # Replace them with custom labels
    new_labels = [create_label(label) for label in yticklabels]
    ax.set_yticklabels(new_labels, fontsize=9)

    for tick, label in zip(yticks, yticklabels):
        count = pos_counts.get(label, 0)
        plt.text(global_max + 0.1, tick, f'n={count}', va='center', ha='left', fontsize=9)

    plt.title(f'Positive Changes Boxplot (Sorted by Count) Part {i}')
    plt.ylabel('Variables')
    plt.xlabel('Standardized Value')
    plt.xlim(global_min, global_max)
    plt.tight_layout()
    plt.savefig(f"Fig/Boxplot_Positive_Sorted_Part{i}.svg", bbox_inches="tight", pad_inches=0.3, format='svg')
    plt.close()

# ---------- NEGATIVE ----------
neg_counts = negative_scaled.notna().sum()
sorted_neg_cols = neg_counts.sort_values(ascending=False).index.tolist()

for i, chunk in enumerate(chunk_list(sorted_neg_cols, 50), start=1):
    chunk_df = negative_scaled[chunk[::-1]]  # Reverse to put higher counts on top

    plt.figure(figsize=(10, max(6, len(chunk) * 0.3)))
    ax = chunk_df.boxplot(vert=False, patch_artist=True, showfliers=False,
                          boxprops=dict(facecolor='lightcoral'))

    # Add count labels
    yticks = ax.get_yticks()
    # Get current tick labels
    yticklabels = [tick.get_text() for tick in ax.get_yticklabels()]
    # Replace them with custom labels
    new_labels = [create_label(label) for label in yticklabels]
    ax.set_yticklabels(new_labels, fontsize=9)

    for tick, label in zip(yticks, yticklabels):
        count = neg_counts.get(label, 0)
        plt.text(global_max + 0.1, tick, f'n={count}', va='center', ha='left', fontsize=9)

    plt.title(f'Negative Changes Boxplot (Sorted by Count) Part {i}')
    plt.ylabel('Variables')
    plt.xlabel('Standardized Value')
    plt.xlim(global_min, global_max)
    plt.tight_layout()
    plt.savefig(f"Fig/Boxplot_Negative_Sorted_Part{i}.svg", bbox_inches="tight", pad_inches=0.3, format='svg')
    plt.close()
