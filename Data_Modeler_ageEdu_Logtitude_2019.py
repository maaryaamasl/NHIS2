import time

import numpy as np
import pandas as pd
import sklearn as sk
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import roc_auc_score
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
# import pycaret as pc
import tpot
from tpot import TPOTClassifier
# import h2o
# from h2o.automl import H2OAutoML
# import autokeras as ak
# from autokeras import StructuredDataClassifier
import shap
# import shapley
from xgboost import XGBClassifier
# from lightgbm import LGBMClassifier
import torch
print(torch.version.cuda)
print(torch.cuda.is_available())


print("\ncleaned_data")
Merged_data = pd.read_csv('Data_longtitude_ageEdu.csv')

print('Merged_data: ', Merged_data.shape)
# Chronic_Pain {0, 1}
# High_impact_chronic_pain {0, 1}
outcomes = ['pain_trajectory']
for column in outcomes:
    print(column, set(Merged_data[column]), Merged_data[column].value_counts().values)
# Outcome <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< VARIABLES & OUTCOMES
print("######### Setting ########" )
outcome = ['pain_trajectory'] # 'Chronic_Pain', 'High_impact_chronic_pain'
filtering = "" # # SEX_A "RACEALLP_A__White"  # "RACEALLP_A__Black/African-American"
val = 1
shap_reason = "pain_trajectory_2019"
print(shap_reason,outcome,filtering,val)
print("######### Filter ###########")
print('cleaned_data: ', Merged_data.shape)
 # & (selected_data['PAIWKLM3M_A'] == 1)
if filtering != "":
    Merged_data = Merged_data[(Merged_data[filtering] == val)]
    Merged_data.drop([filtering], axis=1, inplace=True)
print('cleaned_data: ', Merged_data.shape)

drop_col = [x for x in outcomes if x not in outcome]
print("Outcome:",outcome," \nDropped_col:",drop_col)
Merged_data.drop(drop_col, axis=1, inplace=True) # 'High_impact_chronic_pain'
for column in Merged_data.columns:
    if filtering in column:
        print(column, set(Merged_data[column]))



# print("######### After categorization ###########")

# drop_col = [x for x in outcomes if x not in outcome]
# print("Outcome:",outcome," \nDropped_col:",drop_col)
# cleaned_data.drop(drop_col, axis=1, inplace=True) # 'High_impact_chronic_pain'
# for column in cleaned_data.columns:
#     print(column, set(cleaned_data[column]))

# def get_count_and_percentage(column):
#     count = column.value_counts()
#     percentage = column.value_counts(normalize=True) * 100
#     result = pd.DataFrame({'Count': count, 'Percentage': percentage})
#     return result
#
# for col in cleaned_data.columns:
#     result = get_count_and_percentage(cleaned_data[col])
#     print(f"=== {col} ===")
#     print(result)
#     print("\n")
#
# print("Age mean:", cleaned_data["AGEP_A"].mean())

# Modeling
print("\nModeling")
X = Merged_data.drop(outcome, axis=1)  # Features
# Only keep baseline features (exclude change_ variables)
X = Merged_data.drop(columns=[col for col in Merged_data.columns if col.startswith('change_') or col in outcome])

Y = Merged_data[outcome]  # Target
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)


from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import pandas as pd

# Assuming your data is already split into:
# x_train, x_test, y_train, y_test

# Define models to compare
models = {
    'Logistic Regression': LogisticRegression(),
    'Random Forest': RandomForestClassifier(),
    'XGBoost': XGBClassifier(),
    'SVM': SVC()
}

# Store results
results = []

from sklearn.preprocessing import LabelEncoder

# Encode target labels to numbers
le = LabelEncoder()
y_train_encoded = le.fit_transform(y_train)
y_test_encoded = le.transform(y_test)
Y_encoded = le.fit_transform(Y)

# Train, predict, and evaluate each model
for name, model in models.items():
    model.fit(x_train, y_train_encoded)
    y_pred = model.predict(x_test)
    acc = accuracy_score(y_test_encoded, y_pred)
    results.append({'Model': name, 'Accuracy': acc})

# Create results DataFrame
results_df = pd.DataFrame(results).sort_values(by='Accuracy', ascending=False)
print(results_df)

categorical_non_numeric = ['REGION', 'AGEP_A', 'ORIENT_A', 'MARITAL_A', 'RACEALLP_A', 'EDUC_A', 'MAXEDUC_A',
                           'change_MAXEDUC_A', 'change_EDUC_A', 'change_AGEP_A', 'change_MARITAL_A',
                           ]


# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.utils.data import DataLoader, Dataset
# from sklearn.preprocessing import LabelEncoder, StandardScaler
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score
# import numpy as np
#
# # ===== Assuming your data is already loaded as X and Y =====
# # Standardize numerical features (all your features since one-hot already done)
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)
# x_train, x_test, y_train, y_test = train_test_split(X_scaled, Y, test_size=0.2, random_state=42)
#
# # Encode target labels
# le = LabelEncoder()
# y_train_encoded = le.fit_transform(y_train)
# y_test_encoded = le.transform(y_test)
#
# # ===== Dataset & DataLoader =====
# class TabularDataset(Dataset):
#     def __init__(self, X, y):
#         self.X = torch.tensor(X, dtype=torch.float32)
#         self.y = torch.tensor(y, dtype=torch.long)
#
#     def __len__(self):
#         return len(self.X)
#
#     def __getitem__(self, idx):
#         return self.X[idx], self.y[idx]
#
# train_dataset = TabularDataset(x_train, y_train_encoded)
# test_dataset = TabularDataset(x_test, y_test_encoded)
# train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
# test_loader = DataLoader(test_dataset, batch_size=256)
#
# # ===== FT-Transformer Model =====
# class FTTransformer(nn.Module):
#     def __init__(self, input_dim, num_classes, d_model=128, n_heads=8, num_layers=4, dropout=0.1):
#         super(FTTransformer, self).__init__()
#         # Project input features into d_model dimension
#         self.input_projection = nn.Linear(input_dim, d_model)
#
#         # Transformer encoder layers
#         encoder_layer = nn.TransformerEncoderLayer(
#             d_model=d_model,
#             nhead=n_heads,
#             dropout=dropout,
#             batch_first=True,
#             activation='gelu'
#         )
#         self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
#
#         # Classifier head
#         self.mlp_head = nn.Sequential(
#             nn.LayerNorm(d_model),
#             nn.Linear(d_model, d_model // 2),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#             nn.Linear(d_model // 2, num_classes)
#         )
#
#     def forward(self, x):
#         # Add sequence dimension (treat features as sequence tokens)
#         x = x.unsqueeze(1)  # shape: [batch_size, 1, num_features]
#         x = self.input_projection(x)  # shape: [batch_size, 1, d_model]
#         x = self.transformer_encoder(x)  # shape remains [batch_size, 1, d_model]
#         x = x.squeeze(1)  # remove sequence dimension
#         return self.mlp_head(x)
#
# # ===== Model Instantiation =====
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = FTTransformer(input_dim=X.shape[1], num_classes=len(le.classes_)).to(device)
#
# optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
# criterion = nn.CrossEntropyLoss()
#
# # ===== Training Loop =====
# epochs = 50
# for epoch in range(epochs):
#     model.train()
#     total_loss = 0
#     for X_batch, y_batch in train_loader:
#         X_batch, y_batch = X_batch.to(device), y_batch.to(device)
#         optimizer.zero_grad()
#         outputs = model(X_batch)
#         loss = criterion(outputs, y_batch)
#         loss.backward()
#         optimizer.step()
#         total_loss += loss.item()
#     print(f'Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}')
#
# # ===== Evaluation =====
# model.eval()
# all_preds, all_labels = [], []
# with torch.no_grad():
#     for X_batch, y_batch in test_loader:
#         X_batch, y_batch = X_batch.to(device), y_batch.to(device)
#         outputs = model(X_batch)
#         preds = torch.argmax(outputs, dim=1)
#         all_preds.extend(preds.cpu().numpy())
#         all_labels.extend(y_batch.cpu().numpy())
#
# accuracy = accuracy_score(all_labels, all_preds)
# print(f'\nFT-Transformer Accuracy: {accuracy:.4f}')
# FT-Transformer  81%

"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ===== Assume your data is already preprocessed =====
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
x_train, x_test, y_train, y_test = train_test_split(X_scaled, Y, test_size=0.2, random_state=42)

# Encode target labels
le = LabelEncoder()
y_train_encoded = le.fit_transform(y_train)
y_test_encoded = le.transform(y_test)


# ===== Dataset Class =====
class TabularDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


train_dataset = TabularDataset(x_train, y_train_encoded)
test_dataset = TabularDataset(x_test, y_test_encoded)
train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=256)


# ===== DeepFM Model =====
class DeepFM(nn.Module):
    def __init__(self, input_dim, embedding_dim=8, hidden_layers=[256, 128, 64], dropout=0.3, num_classes=4):
        super(DeepFM, self).__init__()
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim

        # Factorization Machine (FM) part
        self.linear = nn.Linear(input_dim, 1)
        self.feature_embeddings = nn.Parameter(torch.randn(input_dim, embedding_dim))

        # Deep part
        layer_list = []
        current_dim = input_dim * embedding_dim
        for layer_size in hidden_layers:
            layer_list.append(nn.Linear(current_dim, layer_size))
            layer_list.append(nn.ReLU())
            layer_list.append(nn.Dropout(dropout))
            current_dim = layer_size
        self.deep_layers = nn.Sequential(*layer_list)
        self.output_layer = nn.Linear(current_dim + 1, num_classes)  # FM output + deep output

    def forward(self, x):
        # ===== FM part =====
        linear_part = self.linear(x).squeeze(1)  # shape: [batch_size]
        # Second-order interactions (sum-square trick)
        embed_x = torch.matmul(x, self.feature_embeddings)  # [batch_size, embedding_dim]
        square_of_sum = torch.pow(embed_x, 2).sum(dim=1)
        sum_of_square = torch.matmul(x.pow(2), self.feature_embeddings.pow(2)).sum(dim=1)
        interaction_part = 0.5 * (square_of_sum - sum_of_square)  # [batch_size]

        fm_output = linear_part + interaction_part  # [batch_size]

        # ===== Deep part =====
        deep_input = torch.einsum('bi,ij->bij', x, self.feature_embeddings)  # [batch_size, input_dim, embedding_dim]
        deep_input_flat = deep_input.reshape(deep_input.size(0), -1)
        deep_output = self.deep_layers(deep_input_flat)

        # ===== Combine FM + Deep =====
        combined = torch.cat([fm_output.unsqueeze(1), deep_output], dim=1)
        output = self.output_layer(combined)
        return output


# ===== Model, Optimizer, Loss =====
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = DeepFM(input_dim=X.shape[1], num_classes=len(le.classes_)).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

# ===== Training =====
epochs = 50
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f'Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(train_loader):.4f}')

# ===== Evaluation =====
model.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        outputs = model(X_batch)
        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y_batch.cpu().numpy())

accuracy = accuracy_score(all_labels, all_preds)
print(f'\nDeepFM Accuracy: {accuracy:.4f}')

# DeepFM Accuracy: 0.8461
"""

# import pandas as pd
# from autogluon.tabular import TabularPredictor
# from sklearn.preprocessing import LabelEncoder
# from sklearn.model_selection import train_test_split
#
# # ===== Assume your data is loaded as Merged_data and 'pain_trajectory' is your outcome =====
# X = Merged_data.drop('pain_trajectory', axis=1)
# Y = Merged_data['pain_trajectory']
#
# # Encode target labels if they are strings
# le = LabelEncoder()
# Y_encoded = le.fit_transform(Y)
#
# # Combine back into a single dataframe (AutoGluon expects target column inside)
# data = X.copy()
# data['pain_trajectory'] = Y_encoded
#
# # Train-test split
# train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
#
# # Define AutoGluon predictor
# predictor = TabularPredictor(label='pain_trajectory', problem_type='multiclass').fit(
#     train_data=train_data,
#     time_limit=600,  # 10 minutes, adjust as needed
#     presets='best_quality'  # can change to 'medium_quality' or 'fast_train' for speed
# )
#
# # Evaluate on test set
# performance = predictor.evaluate(test_data)
# print(performance)
#
# # View leaderboard of models used
# leaderboard = predictor.leaderboard(test_data, silent=True)
# print(leaderboard)

"""
import torch
print(torch.cuda.is_available())  # Should be True if GPU is working
print(torch.version.cuda)
print(torch.cuda.is_available())  # Should now be True ✅
print(torch.cuda.device_count())  # Number of GPUs detected
print(torch.cuda.get_device_name(0))  # Your GPU name
# exit()

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.preprocessing import StandardScaler

# === Data Preparation ===
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # Assuming your X is ready and numerical/one-hot

class MaskedTabularDataset(Dataset):
    def __init__(self, X, mask_prob=0.3):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.mask_prob = mask_prob

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        original = self.X[idx]
        mask = (torch.rand(original.shape) > self.mask_prob).float()
        masked_input = original * mask
        return masked_input, original, mask  # input, target, mask for loss calculation

train_dataset = MaskedTabularDataset(X_scaled)
train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)

# === Simple MLP Autoencoder ===
class TabularAutoencoder(nn.Module):
    def __init__(self, input_dim):
        super(TabularAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

model = TabularAutoencoder(input_dim=X.shape[1]).cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss(reduction='none')  # We'll apply the mask manually

# === Pretraining Loop ===
epochs = 100
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for X_masked, X_orig, mask in train_loader:
        X_masked, X_orig, mask = X_masked.cuda(), X_orig.cuda(), mask.cuda()
        optimizer.zero_grad()
        outputs = model(X_masked)
        loss = (criterion(outputs, X_orig) * (1 - mask)).sum() / (1 - mask).sum()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f'Epoch {epoch+1}/{epochs}, Pretraining Loss: {total_loss/len(train_loader):.4f}')


# === Classifier Model (Using Pretrained Encoder) ===
class TabularClassifier(nn.Module):
    def __init__(self, pretrained_encoder, num_classes):
        super(TabularClassifier, self).__init__()
        self.encoder = pretrained_encoder.encoder  # Use the pretrained encoder part only
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),  # Output size of encoder
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        return self.classifier(encoded)

# === Prepare your supervised dataset (Pain Trajectory) ===
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

# Target encoding
le = LabelEncoder()
Y_encoded = le.fit_transform(Y)

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(X_scaled, Y_encoded, test_size=0.2, random_state=42)

# Supervised Dataset class
class SupervisedDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_dataset = SupervisedDataset(x_train, y_train)
test_dataset = SupervisedDataset(x_test, y_test)
train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=256)

# === Instantiate classifier ===
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
classifier_model = TabularClassifier(pretrained_encoder=model, num_classes=len(le.classes_)).to(device)
optimizer = torch.optim.AdamW(classifier_model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

# === Fine-tuning Loop ===
epochs = 50
for epoch in range(epochs):
    classifier_model.train()
    total_loss = 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        outputs = classifier_model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f'Fine-tuning Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}')

# === Evaluation ===
from sklearn.metrics import accuracy_score, classification_report

classifier_model.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        outputs = classifier_model(X_batch)
        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y_batch.cpu().numpy())

accuracy = accuracy_score(all_labels, all_preds)
print(f'\nAccuracy after Self-Supervised Pretraining + Fine-tuning: {accuracy:.4f}')
print("\nClassification Report:")
print(classification_report(all_labels, all_preds, target_names=le.classes_.astype(str)))

# Accuracy after Self-Supervised Pretraining + Fine-tuning: 0.8256
"""
#
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.utils.data import Dataset, DataLoader
# import numpy as np
# from sklearn.preprocessing import StandardScaler
#
# # Original baseline features (e.g., 2019)
# X_baseline = Merged_data[[col for col in Merged_data.columns if not col.startswith('change_') and col != 'pain_trajectory']]
#
# # Changes (delta features)
# X_change = Merged_data[[col for col in Merged_data.columns if col.startswith('change_')]]
#
# Y = Merged_data['pain_trajectory']
#
# # Standardize both separately
# scaler_baseline = StandardScaler()
# scaler_change = StandardScaler()
# X_baseline_scaled = scaler_baseline.fit_transform(X_baseline)
# X_change_scaled = scaler_change.fit_transform(X_change)
#
# # Target encoding
# le = LabelEncoder()
# Y_encoded = le.fit_transform(Y)
#
# # Train-test split
# x_train_base, x_test_base, x_train_change, x_test_change, y_train, y_test = train_test_split(
#     X_baseline_scaled, X_change_scaled, Y_encoded, test_size=0.2, random_state=42)
# class BaselineChangeDataset(Dataset):
#     def __init__(self, X_base, X_change, y):
#         self.X_base = torch.tensor(X_base, dtype=torch.float32)
#         self.X_change = torch.tensor(X_change, dtype=torch.float32)
#         self.y = torch.tensor(y, dtype=torch.long)
#     def __len__(self):
#         return len(self.X_base)
#     def __getitem__(self, idx):
#         return self.X_base[idx], self.X_change[idx], self.y[idx]
#
# train_dataset = BaselineChangeDataset(x_train_base, x_train_change, y_train)
# test_dataset = BaselineChangeDataset(x_test_base, x_test_change, y_test)
# train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
# test_loader = DataLoader(test_dataset, batch_size=256)
# class BaselineChangeAttentionModel(nn.Module):
#     def __init__(self, input_dim_base, input_dim_change, d_model=128, n_heads=4, num_layers=2, num_classes=4):
#         super(BaselineChangeAttentionModel, self).__init__()
#         # Separate projections for baseline and change
#         self.proj_base = nn.Linear(input_dim_base, d_model)
#         self.proj_change = nn.Linear(input_dim_change, d_model)
#
#         # Transformer encoder
#         encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, batch_first=True)
#         self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
#
#         # Classifier head
#         self.classifier = nn.Sequential(
#             nn.LayerNorm(d_model),
#             nn.Linear(d_model, d_model // 2),
#             nn.ReLU(),
#             nn.Dropout(0.3),
#             nn.Linear(d_model // 2, num_classes)
#         )
#
#     def forward(self, x_base, x_change):
#         token_base = self.proj_base(x_base).unsqueeze(1)      # [batch, 1, d_model]
#         token_change = self.proj_change(x_change).unsqueeze(1)  # [batch, 1, d_model]
#         tokens = torch.cat([token_base, token_change], dim=1)   # [batch, 2, d_model]
#         encoded = self.transformer(tokens)
#         pooled = encoded.mean(dim=1)
#         return self.classifier(pooled)
# # ===== Instantiate model =====
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = BaselineChangeAttentionModel(
#     input_dim_base=X_baseline.shape[1],
#     input_dim_change=X_change.shape[1],
#     num_classes=len(le.classes_)
# ).to(device)
#
# optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
# criterion = nn.CrossEntropyLoss()
#
# # ===== Training Loop =====
# epochs = 50
# for epoch in range(epochs):
#     model.train()
#     total_loss = 0
#     for x_base, x_change, y_batch in train_loader:
#         x_base, x_change, y_batch = x_base.to(device), x_change.to(device), y_batch.to(device)
#         optimizer.zero_grad()
#         outputs = model(x_base, x_change)
#         loss = criterion(outputs, y_batch)
#         loss.backward()
#         optimizer.step()
#         total_loss += loss.item()
#     print(f'Epoch {epoch+1}/{epochs}, Training Loss: {total_loss/len(train_loader):.4f}')
#
# # ===== Evaluation =====
# from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# import seaborn as sns
# import matplotlib.pyplot as plt
#
# model.eval()
# all_preds, all_labels = [], []
# with torch.no_grad():
#     for x_base, x_change, y_batch in test_loader:
#         x_base, x_change, y_batch = x_base.to(device), x_change.to(device), y_batch.to(device)
#         outputs = model(x_base, x_change)
#         preds = torch.argmax(outputs, dim=1)
#         all_preds.extend(preds.cpu().numpy())
#         all_labels.extend(y_batch.cpu().numpy())
#
# accuracy = accuracy_score(all_labels, all_preds)
# print(f'\nAccuracy (Baseline + Change Attention Model): {accuracy:.4f}')
#
# # Classification report
# print("\nClassification Report:")
# print(classification_report(all_labels, all_preds, target_names=le.classes_.astype(str)))
#
# # Optional: Confusion matrix visualization
# cm = confusion_matrix(all_labels, all_preds)
# plt.figure(figsize=(8, 6))
# sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
# plt.xlabel('Predicted')
# plt.ylabel('True')
# plt.title('Confusion Matrix')
# plt.show()
# # Accuracy (Baseline + Change Attention Model): 0.8373

"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
# Assuming your merged data is in 'Merged_data'
Y = Merged_data['pain_trajectory']
group_col = 'RACEALLP_A__White'  # CHANGE THIS to your demographic column (e.g., 'gender' or 'race')

X = Merged_data.drop(['pain_trajectory'], axis=1).drop(columns=[group_col], errors='ignore')
groups = Merged_data[group_col]

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Encode labels
le = LabelEncoder()
Y_encoded = le.fit_transform(Y)
group_encoder = LabelEncoder()
groups_encoded = group_encoder.fit_transform(groups)

# Train-test split
x_train, x_test, y_train, y_test, group_train, group_test = train_test_split(
    X_scaled, Y_encoded, groups_encoded, test_size=0.2, random_state=42)

class GroupAwareDataset(Dataset):
    def __init__(self, X, y, groups):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        self.groups = torch.tensor(groups, dtype=torch.long)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.groups[idx]

batch_size = 256
train_loader = DataLoader(GroupAwareDataset(x_train, y_train, group_train), batch_size=batch_size, shuffle=True)
test_loader = DataLoader(GroupAwareDataset(x_test, y_test, group_test), batch_size=batch_size)

def group_contrastive_loss(embeddings, labels, groups, temperature=0.5):
    batch_size = embeddings.shape[0]
    labels = labels.unsqueeze(1)
    groups = groups.unsqueeze(1)

    sim_matrix = F.cosine_similarity(embeddings.unsqueeze(1), embeddings.unsqueeze(0), dim=2)
    positive_mask = (labels == labels.T) & (groups != groups.T)
    negative_mask = ~positive_mask

    positives = sim_matrix * positive_mask.float()
    numerator = torch.exp(positives / temperature).sum(dim=1)
    denominator = torch.exp(sim_matrix / temperature).sum(dim=1) - torch.exp(torch.ones(batch_size, device=embeddings.device) / temperature)#torch.exp(torch.ones(batch_size) / temperature)
    loss = -torch.log((numerator + 1e-8) / (denominator + 1e-8))
    return loss.mean()
class GroupAwareEncoder(nn.Module):
    def __init__(self, input_dim, embedding_dim=128, num_classes=4, num_groups=3):
        super(GroupAwareEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, embedding_dim)
        )
        self.classifier = nn.Linear(embedding_dim, num_classes)
        self.group_classifier = nn.Linear(embedding_dim, num_groups)

    def forward(self, x):
        embedding = self.encoder(x)
        embedding = F.normalize(embedding, dim=1)
        class_logits = self.classifier(embedding)
        group_logits = self.group_classifier(embedding)
        return embedding, class_logits, group_logits
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GroupAwareEncoder(input_dim=X.shape[1], num_classes=len(le.classes_), num_groups=len(np.unique(groups_encoded))).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
classification_loss_fn = nn.CrossEntropyLoss()
group_loss_fn = nn.CrossEntropyLoss()

epochs = 50
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for X_batch, y_batch, group_batch in train_loader:
        X_batch, y_batch, group_batch = X_batch.to(device), y_batch.to(device), group_batch.to(device)
        optimizer.zero_grad()
        embeddings, class_logits, group_logits = model(X_batch)
        supervised_loss = classification_loss_fn(class_logits, y_batch)
        contrastive_loss = group_contrastive_loss(embeddings, y_batch, group_batch)
        adversarial_loss = group_loss_fn(group_logits, group_batch)
        loss = supervised_loss + 0.5 * contrastive_loss + 0.1 * adversarial_loss  # You can tune these weights
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f'Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}')
model.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for X_batch, y_batch, _ in test_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        embeddings, class_logits, _ = model(X_batch)
        preds = torch.argmax(class_logits, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y_batch.cpu().numpy())

accuracy = accuracy_score(all_labels, all_preds)
print(f'\nFinal Accuracy (GSSL): {accuracy:.4f}')
print("\nClassification Report:")
print(classification_report(all_labels, all_preds, target_names=le.classes_.astype(str)))
# Final Accuracy (GSSL): 0.8013
"""

# # ===== Imports =====
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.utils.data import Dataset, DataLoader
# from sklearn.preprocessing import LabelEncoder, StandardScaler
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score, classification_report
# import numpy as np
# import pandas as pd
#
# # ===== Data Preparation =====
# Y = Merged_data['pain_trajectory']
# group_col = 'RACEALLP_A'  # Optional for later extensions
#
# X = Merged_data.drop(['pain_trajectory'], axis=1).drop(columns=[group_col], errors='ignore')
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)
#
# le = LabelEncoder()
# Y_encoded = le.fit_transform(Y)
#
# x_train, x_test, y_train, y_test = train_test_split(X_scaled, Y_encoded, test_size=0.2, random_state=42)
#
# class TabularDataset(Dataset):
#     def __init__(self, X, y):
#         self.X = torch.tensor(X, dtype=torch.float32)
#         self.y = torch.tensor(y, dtype=torch.long)
#     def __len__(self):
#         return len(self.X)
#     def __getitem__(self, idx):
#         return self.X[idx], self.y[idx]
#
# batch_size = 256
# train_loader = DataLoader(TabularDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
# test_loader = DataLoader(TabularDataset(x_test, y_test), batch_size=batch_size)
#
# # ===== VAE Model with Classifier Head =====
# class VAEContrastive(nn.Module):
#     def __init__(self, input_dim, latent_dim=32, num_classes=4):
#         super(VAEContrastive, self).__init__()
#         self.encoder = nn.Sequential(
#             nn.Linear(input_dim, 256),
#             nn.ReLU(),
#             nn.Linear(256, 128),
#             nn.ReLU()
#         )
#         self.fc_mu = nn.Linear(128, latent_dim)
#         self.fc_logvar = nn.Linear(128, latent_dim)
#         self.decoder = nn.Sequential(
#             nn.Linear(latent_dim, 128),
#             nn.ReLU(),
#             nn.Linear(128, 256),
#             nn.ReLU(),
#             nn.Linear(256, input_dim)
#         )
#         self.classifier = nn.Linear(latent_dim, num_classes)
#
#     def reparameterize(self, mu, logvar):
#         std = torch.exp(0.5 * logvar)
#         eps = torch.randn_like(std)
#         return mu + eps * std
#
#     def forward(self, x):
#         enc = self.encoder(x)
#         mu = self.fc_mu(enc)
#         logvar = self.fc_logvar(enc)
#         z = self.reparameterize(mu, logvar)
#         recon = self.decoder(z)
#         logits = self.classifier(z)
#         return recon, mu, logvar, z, logits
#
# # ===== Contrastive Loss (SupCon-like) =====
# def contrastive_loss(z, labels, temperature=0.5):
#     z = F.normalize(z, dim=1)
#     sim_matrix = torch.matmul(z, z.T) / temperature
#     labels = labels.unsqueeze(1)
#     mask = torch.eq(labels, labels.T).float()
#     logits = sim_matrix - torch.eye(z.shape[0], device=z.device) * 1e12  # remove self-similarity
#     exp_logits = torch.exp(logits)
#     log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True))
#     loss = -(mask * log_prob).sum(1) / mask.sum(1)
#     return loss.mean()
#
# # ===== VAE Loss =====
# def vae_loss_function(recon_x, x, mu, logvar):
#     recon_loss = F.mse_loss(recon_x, x, reduction='mean')
#     kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
#     return recon_loss + kl_div
#
# # ===== Training Loop =====
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = VAEContrastive(input_dim=X.shape[1], num_classes=len(le.classes_)).to(device)
# optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
# classification_loss_fn = nn.CrossEntropyLoss()
#
# epochs = 50
# for epoch in range(epochs):
#     model.train()
#     total_loss = 0
#     for X_batch, y_batch in train_loader:
#         X_batch, y_batch = X_batch.to(device), y_batch.to(device)
#         optimizer.zero_grad()
#         recon, mu, logvar, z, logits = model(X_batch)
#         vae_loss = vae_loss_function(recon, X_batch, mu, logvar)
#         clf_loss = classification_loss_fn(logits, y_batch)
#         contr_loss = contrastive_loss(z, y_batch)
#         loss = vae_loss + clf_loss + 0.5 * contr_loss  # Tune contrastive weight as needed
#         loss.backward()
#         optimizer.step()
#         total_loss += loss.item()
#     print(f'Epoch {epoch+1}/{epochs}, Total Loss: {total_loss/len(train_loader):.4f}')
#
# # ===== Evaluation =====
# model.eval()
# all_preds, all_labels = [], []
# with torch.no_grad():
#     for X_batch, y_batch in test_loader:
#         X_batch, y_batch = X_batch.to(device), y_batch.to(device)
#         _, _, _, z, logits = model(X_batch)
#         preds = torch.argmax(logits, dim=1)
#         all_preds.extend(preds.cpu().numpy())
#         all_labels.extend(y_batch.cpu().numpy())
#
# accuracy = accuracy_score(all_labels, all_preds)
# print(f'\nFinal Accuracy (VAE + Contrastive): {accuracy:.4f}')
# print("\nClassification Report:")
# print(classification_report(all_labels, all_preds, target_names=le.classes_.astype(str)))
# # Final Accuracy (VAE + Contrastive): 0.8568

"""
# ===== Imports =====
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
import pandas as pd

# ===== Data Preparation =====
Y = Merged_data['pain_trajectory']
group_col = 'RACEALLP_A'  # Optional for fairness later

X = Merged_data.drop(['pain_trajectory'], axis=1).drop(columns=[group_col], errors='ignore')
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

le = LabelEncoder()
Y_encoded = le.fit_transform(Y)

x_train, x_test, y_train, y_test = train_test_split(X_scaled, Y_encoded, test_size=0.2, random_state=42)

class TabularDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

batch_size = 256
train_loader = DataLoader(TabularDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
test_loader = DataLoader(TabularDataset(x_test, y_test), batch_size=batch_size)

# ===== Attention-based VAE Encoder with Contrastive Learning =====
class AttentionVAEContrastive(nn.Module):
    def __init__(self, input_dim, latent_dim=32, num_classes=4):
        super(AttentionVAEContrastive, self).__init__()
        self.feature_attention = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.Sigmoid()  # Gating attention over features
        )
        self.encoder_base = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(128, latent_dim)
        self.fc_logvar = nn.Linear(128, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim)
        )
        self.classifier = nn.Linear(latent_dim, num_classes)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        attention_weights = self.feature_attention(x)
        x_att = x * attention_weights  # Apply feature-wise attention
        enc = self.encoder_base(x_att)
        mu = self.fc_mu(enc)
        logvar = self.fc_logvar(enc)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z)
        logits = self.classifier(z)
        return recon, mu, logvar, z, logits, attention_weights

# ===== Contrastive Loss =====
def contrastive_loss(z, labels, temperature=0.5):
    z = F.normalize(z, dim=1)
    sim_matrix = torch.matmul(z, z.T) / temperature
    labels = labels.unsqueeze(1)
    mask = torch.eq(labels, labels.T).float()
    logits = sim_matrix - torch.eye(z.shape[0], device=z.device) * 1e12
    exp_logits = torch.exp(logits)
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True))
    loss = -(mask * log_prob).sum(1) / mask.sum(1)
    return loss.mean()

# ===== VAE Loss =====
def vae_loss_function(recon_x, x, mu, logvar):
    recon_loss = F.mse_loss(recon_x, x, reduction='mean')
    kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
    return recon_loss + kl_div

# ===== Training Loop =====
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = AttentionVAEContrastive(input_dim=X.shape[1], num_classes=len(le.classes_)).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
classification_loss_fn = nn.CrossEntropyLoss()

epochs = 50
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        recon, mu, logvar, z, logits, attn_weights = model(X_batch)
        vae_loss = vae_loss_function(recon, X_batch, mu, logvar)
        clf_loss = classification_loss_fn(logits, y_batch)
        contr_loss = contrastive_loss(z, y_batch)
        loss = vae_loss + clf_loss + 0.5 * contr_loss  # Tune contrastive weight if needed
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f'Epoch {epoch+1}/{epochs}, Total Loss: {total_loss/len(train_loader):.4f}')

# ===== Evaluation =====
model.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        _, _, _, z, logits, attn_weights = model(X_batch)
        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y_batch.cpu().numpy())

accuracy = accuracy_score(all_labels, all_preds)
print(f'\nFinal Accuracy (Attention VAE + Contrastive): {accuracy:.4f}')
print("\nClassification Report:")
print(classification_report(all_labels, all_preds, target_names=le.classes_.astype(str)))
"""
# # ===== Imports =====
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.utils.data import Dataset, DataLoader
# from sklearn.preprocessing import LabelEncoder, StandardScaler
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score, classification_report
# import numpy as np
# import pandas as pd
#
# # ===== Data Preparation =====
# Y = Merged_data['pain_trajectory']
# group_col = 'RACEALLP_A'  # Optional for fairness later
#
# X = Merged_data.drop(['pain_trajectory'], axis=1).drop(columns=[group_col], errors='ignore')
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)
#
# le = LabelEncoder()
# Y_encoded = le.fit_transform(Y)
#
# x_train, x_test, y_train, y_test = train_test_split(X_scaled, Y_encoded, test_size=0.2, random_state=42)
#
#
# class TabularDataset(Dataset):
#     def __init__(self, X, y):
#         self.X = torch.tensor(X, dtype=torch.float32)
#         self.y = torch.tensor(y, dtype=torch.long)
#
#     def __len__(self):
#         return len(self.X)
#
#     def __getitem__(self, idx):
#         return self.X[idx], self.y[idx]
#
#
# batch_size = 256
# train_loader = DataLoader(TabularDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
# test_loader = DataLoader(TabularDataset(x_test, y_test), batch_size=batch_size)
#
#
# # ===== Prototype Network =====
# class PrototypeNet(nn.Module):
#     def __init__(self, input_dim, latent_dim=64, num_classes=4):
#         super(PrototypeNet, self).__init__()
#         self.encoder = nn.Sequential(
#             nn.Linear(input_dim, 256),
#             nn.ReLU(),
#             nn.Linear(256, latent_dim)
#         )
#         # Learnable prototypes (1 per class)
#         self.prototypes = nn.Parameter(torch.randn(num_classes, latent_dim))
#
#     def forward(self, x):
#         embedding = self.encoder(x)
#         embedding = F.normalize(embedding, dim=1)
#         proto_norm = F.normalize(self.prototypes, dim=1)
#         # Compute distances to prototypes (Euclidean or cosine)
#         distances = torch.cdist(embedding.unsqueeze(1), proto_norm.unsqueeze(0)).squeeze(1)
#         # Use negative distance as logits
#         logits = -distances
#         return logits, embedding
#
#
# # ===== Training =====
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = PrototypeNet(input_dim=X.shape[1], num_classes=len(le.classes_)).to(device)
# optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
# loss_fn = nn.CrossEntropyLoss()
#
# epochs = 100
# for epoch in range(epochs):
#     model.train()
#     total_loss = 0
#     for X_batch, y_batch in train_loader:
#         X_batch, y_batch = X_batch.to(device), y_batch.to(device)
#         optimizer.zero_grad()
#         logits, embeddings = model(X_batch)
#         loss = loss_fn(logits, y_batch)
#         loss.backward()
#         optimizer.step()
#         total_loss += loss.item()
#     print(f'Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(train_loader):.4f}')
#
# # ===== Evaluation =====
# model.eval()
# all_preds, all_labels = [], []
# with torch.no_grad():
#     for X_batch, y_batch in test_loader:
#         X_batch, y_batch = X_batch.to(device), y_batch.to(device)
#         logits, _ = model(X_batch)
#         preds = torch.argmax(logits, dim=1)
#         all_preds.extend(preds.cpu().numpy())
#         all_labels.extend(y_batch.cpu().numpy())
#
# accuracy = accuracy_score(all_labels, all_preds)
# print(f'\nFinal Accuracy (Prototype Network): {accuracy:.4f}')
# print("\nClassification Report:")
# print(classification_report(all_labels, all_preds, target_names=le.classes_.astype(str)))
"""

# ===== Imports =====
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
import pandas as pd

# ===== Data Preparation =====
Y = Merged_data['pain_trajectory']
group_col = 'RACEALLP_A'  # Optional

baseline_cols = [col for col in Merged_data.columns if not col.startswith('change_') and col not in ['pain_trajectory', group_col]]
change_cols = [col for col in Merged_data.columns if col.startswith('change_')]

X_baseline = Merged_data[baseline_cols]
X_change = Merged_data[change_cols]

scaler_base = StandardScaler()
X_baseline_scaled = scaler_base.fit_transform(X_baseline)
scaler_change = StandardScaler()
X_change_scaled = scaler_change.fit_transform(X_change)

le = LabelEncoder()
Y_encoded = le.fit_transform(Y)

x_base_train, x_base_test, x_change_train, x_change_test, y_train, y_test = train_test_split(
    X_baseline_scaled, X_change_scaled, Y_encoded, test_size=0.2, random_state=42)

class PairedDataset(Dataset):
    def __init__(self, X_base, X_change, y):
        self.X_base = torch.tensor(X_base, dtype=torch.float32)
        self.X_change = torch.tensor(X_change, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self):
        return len(self.X_base)
    def __getitem__(self, idx):
        return self.X_base[idx], self.X_change[idx], self.y[idx]

batch_size = 256
train_loader = DataLoader(PairedDataset(x_base_train, x_change_train, y_train), batch_size=batch_size, shuffle=True)
test_loader = DataLoader(PairedDataset(x_base_test, x_change_test, y_test), batch_size=batch_size)

# ===== Cross-Attention Model =====
class CrossAttentionModel(nn.Module):
    def __init__(self, baseline_dim, change_dim, hidden_dim=128, num_classes=4, n_heads=4):
        super(CrossAttentionModel, self).__init__()
        self.baseline_proj = nn.Linear(baseline_dim, hidden_dim)
        self.change_proj = nn.Linear(change_dim, hidden_dim)
        self.attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=n_heads, batch_first=True)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, baseline, change):
        # [batch_size, features] -> [batch_size, 1, hidden_dim]
        baseline = self.baseline_proj(baseline).unsqueeze(1)
        change = self.change_proj(change).unsqueeze(1)
        # Query: baseline, Key/Value: change (cross-attention)
        attended_output, _ = self.attention(query=baseline, key=change, value=change)
        attended_output = attended_output.squeeze(1)
        logits = self.classifier(attended_output)
        return logits

# ===== Training Setup =====
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CrossAttentionModel(baseline_dim=X_baseline.shape[1], change_dim=X_change.shape[1], num_classes=len(le.classes_)).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

epochs = 50
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for X_base_batch, X_change_batch, y_batch in train_loader:
        X_base_batch, X_change_batch, y_batch = X_base_batch.to(device), X_change_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        logits = model(X_base_batch, X_change_batch)
        loss = loss_fn(logits, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f'Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}')

# ===== Evaluation =====
model.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for X_base_batch, X_change_batch, y_batch in test_loader:
        X_base_batch, X_change_batch, y_batch = X_base_batch.to(device), X_change_batch.to(device), y_batch.to(device)
        logits = model(X_base_batch, X_change_batch)
        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y_batch.cpu().numpy())

accuracy = accuracy_score(all_labels, all_preds)
print(f'\nFinal Accuracy (Cross-Attention Model): {accuracy:.4f}')
print("\nClassification Report:")
print(classification_report(all_labels, all_preds, target_names=le.classes_.astype(str)))


"""
#
# # ===== Imports =====
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.utils.data import Dataset, DataLoader
# from sklearn.preprocessing import LabelEncoder, StandardScaler
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score, classification_report
# import numpy as np
# import pandas as pd
#
# # ===== Data Preparation =====
# Y = Merged_data['pain_trajectory']
# group_col = 'RACEALLP_A__White'  # Optional demographic info
#
# baseline_cols = [col for col in Merged_data.columns if not col.startswith('change_') and col not in ['pain_trajectory', group_col]]
# change_cols = [col for col in Merged_data.columns if col.startswith('change_')]
# demo_cols = [group_col]
#
# X_baseline = Merged_data[baseline_cols]
# X_change = Merged_data[change_cols]
# X_demo = Merged_data[demo_cols]
#
# scaler_base = StandardScaler()
# X_baseline_scaled = scaler_base.fit_transform(X_baseline)
# scaler_change = StandardScaler()
# X_change_scaled = scaler_change.fit_transform(X_change)
# scaler_demo = StandardScaler()
# X_demo_scaled = scaler_demo.fit_transform(X_demo)
#
# le = LabelEncoder()
# Y_encoded = le.fit_transform(Y)
#
# x_base_train, x_base_test, x_change_train, x_change_test, x_demo_train, x_demo_test, y_train, y_test = train_test_split(
#     X_baseline_scaled, X_change_scaled, X_demo_scaled, Y_encoded, test_size=0.2, random_state=42)
#
# class MultiViewDataset(Dataset):
#     def __init__(self, X_base, X_change, X_demo, y):
#         self.X_base = torch.tensor(X_base, dtype=torch.float32)
#         self.X_change = torch.tensor(X_change, dtype=torch.float32)
#         self.X_demo = torch.tensor(X_demo, dtype=torch.float32)
#         self.y = torch.tensor(y, dtype=torch.long)
#     def __len__(self):
#         return len(self.X_base)
#     def __getitem__(self, idx):
#         return self.X_base[idx], self.X_change[idx], self.X_demo[idx], self.y[idx]
#
# batch_size = 256
# train_loader = DataLoader(MultiViewDataset(x_base_train, x_change_train, x_demo_train, y_train), batch_size=batch_size, shuffle=True)
# test_loader = DataLoader(MultiViewDataset(x_base_test, x_change_test, x_demo_test, y_test), batch_size=batch_size)
#
# # ===== MixUp Augmentation Function =====
# def mixup_data(x1, x2, x3, y, alpha=0.4):
#     lam = np.random.beta(alpha, alpha)
#     batch_size = x1.size(0)
#     index = torch.randperm(batch_size)
#     x1_mix = lam * x1 + (1 - lam) * x1[index]
#     x2_mix = lam * x2 + (1 - lam) * x2[index]
#     x3_mix = lam * x3 + (1 - lam) * x3[index]
#     y_mix = lam * y + (1 - lam) * y[index]
#     return x1_mix, x2_mix, x3_mix, y_mix.long()
#
# # ===== Multi-View Fusion Model =====
# class MultiViewFusion(nn.Module):
#     def __init__(self, base_dim, change_dim, demo_dim, hidden_dim=128, num_classes=4):
#         super(MultiViewFusion, self).__init__()
#         self.base_encoder = nn.Sequential(nn.Linear(base_dim, hidden_dim), nn.ReLU())
#         self.change_encoder = nn.Sequential(nn.Linear(change_dim, hidden_dim), nn.ReLU())
#         self.demo_encoder = nn.Sequential(nn.Linear(demo_dim, hidden_dim), nn.ReLU())
#         self.fusion = nn.Sequential(
#             nn.Linear(hidden_dim * 3, 256),
#             nn.ReLU(),
#             nn.Linear(256, num_classes)
#         )
#
#     def forward(self, base, change, demo):
#         base_out = self.base_encoder(base)
#         change_out = self.change_encoder(change)
#         demo_out = self.demo_encoder(demo)
#         combined = torch.cat([base_out, change_out, demo_out], dim=1)
#         return self.fusion(combined)
#
# # ===== Training Setup =====
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = MultiViewFusion(base_dim=X_baseline.shape[1], change_dim=X_change.shape[1], demo_dim=X_demo.shape[1], num_classes=len(le.classes_)).to(device)
# optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
# loss_fn = nn.CrossEntropyLoss()
#
# # ===== Training Loop with MixUp =====
# epochs = 50
# for epoch in range(epochs):
#     model.train()
#     total_loss = 0
#     for X_base_batch, X_change_batch, X_demo_batch, y_batch in train_loader:
#         X_base_batch, X_change_batch, X_demo_batch, y_batch = X_base_batch.to(device), X_change_batch.to(device), X_demo_batch.to(device), y_batch.to(device)
#         X_base_batch, X_change_batch, X_demo_batch, y_batch = mixup_data(X_base_batch, X_change_batch, X_demo_batch, y_batch)
#         optimizer.zero_grad()
#         logits = model(X_base_batch, X_change_batch, X_demo_batch)
#         loss = loss_fn(logits, y_batch)
#         loss.backward()
#         optimizer.step()
#         total_loss += loss.item()
#     print(f'Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}')
#
# # ===== Evaluation =====
# model.eval()
# all_preds, all_labels = [], []
# with torch.no_grad():
#     for X_base_batch, X_change_batch, X_demo_batch, y_batch in test_loader:
#         X_base_batch, X_change_batch, X_demo_batch, y_batch = X_base_batch.to(device), X_change_batch.to(device), X_demo_batch.to(device), y_batch.to(device)
#         logits = model(X_base_batch, X_change_batch, X_demo_batch)
#         preds = torch.argmax(logits, dim=1)
#         all_preds.extend(preds.cpu().numpy())
#         all_labels.extend(y_batch.cpu().numpy())
#
# accuracy = accuracy_score(all_labels, all_preds)
# print(f'\nFinal Accuracy (Multi-View Fusion + MixUp): {accuracy:.4f}')
# print("\nClassification Report:")
# print(classification_report(all_labels, all_preds, target_names=le.classes_.astype(str)))



# exit()


### Auto ML ###
# tpot
# pycaret
# h2o
# auto-sklearn
# autokeras
# autogluon
# Hyperopt-Sklearn
# Auto-ViML
# MLBox

"""
print("h2o")  ############################### <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
h2o.init(max_mem_size="8G")
aml = H2OAutoML(max_models=20, seed=1, sort_metric = "accuracy") # before 20 eresult below
x=X.columns.tolist()
y=Y.columns.tolist()[0]
cleaned_data_h2o= h2o.H2OFrame(Merged_data)
cleaned_data_h2o[y] = cleaned_data_h2o[y].asfactor()
print(len(x),x,"\n",y)
train, test = cleaned_data_h2o.split_frame(ratios=[0.8], seed=1)
aml.train(x=x, y=y, training_frame=train)
leader_model = aml.leader
predictions = leader_model.predict(test)
accuracy = leader_model.model_performance(test).accuracy()
print(f"Accuracy of the leader model: {accuracy}")
leaderboard = aml.leaderboard
print(leaderboard)
leaderboard_all_metrics = aml.leaderboard.as_data_frame()
print(leaderboard_all_metrics)
for model_id in leaderboard['model_id']:
    model = h2o.get_model(model_id)
    accuracy = model.model_performance(test).accuracy()
    print(f"Accuracy for {model_id}: {accuracy}")
print(outcome)
exit(1)"""

# print("h2o")  ############################### <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# h2o.init(max_mem_size="8G")
# aml = H2OAutoML(max_models=20, seed=1, sort_metric = "accuracy") # before 20 eresult below
# x=X.columns.tolist()
# y=Y.columns.tolist()[0]
# cleaned_data_h2o= h2o.H2OFrame(Merged_data)
# cleaned_data_h2o[y] = cleaned_data_h2o[y].asfactor()
# print(len(x),x,"\n",y)
# train, test = cleaned_data_h2o.split_frame(ratios=[0.8], seed=1)
# aml.train(x=x, y=y, training_frame=train)
# leader_model = aml.leader
# predictions = leader_model.predict(test)
# accuracy = leader_model.model_performance(test).accuracy()
# print(f"Accuracy of the leader model: {accuracy}")
# leaderboard = aml.leaderboard
# print(leaderboard)
# leaderboard_all_metrics = aml.leaderboard.as_data_frame()
# print(leaderboard_all_metrics)
# for model_id in leaderboard['model_id']:
#     model = h2o.get_model(model_id)
#     accuracy = model.model_performance(test).accuracy()
#     print(f"Accuracy for {model_id}: {accuracy}")
# print(outcome)
# exit(1)





print("XGboost") ############################### <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< Chosen model
# x_train, y_train_encoded, x_test, y_test_encoded, y_pred

clf = XGBClassifier()
clf.fit(X, Y_encoded)
y_pred_clf = clf.predict(X)
accuracy = accuracy_score(Y_encoded, y_pred_clf)
y_pred_proba = clf.predict_proba(X)
auc = roc_auc_score(Y_encoded, y_pred_proba, multi_class='ovr')
print("Accuracy (on full data):", outcome, accuracy, auc)
# ['High_impact_chronic_pain'] 0.9716733806509368 0.8425071151403117
# ['High_impact_chronic_pain'] 0.9234335038363172 0.6440411875664475
# ['Chronic_Pain'] 0.8935673636421766 0.7915005806795882
# ['Chronic_Pain'] 0.835485933503836 0.66474041732368
# exit(-1)


exit()

def custom_predict(X):
    return clf.predict_proba(X)
kmeans_k =100 # 100
rows_devideby_to_use = 1 # 1
explainer = shap.KernelExplainer(custom_predict, shap.kmeans(X.values, kmeans_k))
number_of_rows = X.values.shape[0]
print("Size to explain: ",number_of_rows//rows_devideby_to_use)
random_indices = np.random.choice(number_of_rows, size=number_of_rows//rows_devideby_to_use, replace=False)
random_rows = X.iloc[random_indices] #.values
print("explainer.shap_values")
shap_values = explainer.shap_values(random_rows)

print('training-ish size:', len(random_rows.values), len(random_rows.values[0]))
print('\nD1 Classes:', len(shap_values), '\nD2 samples:', len(shap_values[0]))#, '\nD3 Columns/features:', len(shap_values[0][0])) # , '\nvalue:', shap_values[0][0][0]
print('type: ',type(shap_values))
print('type [0]: ', type(shap_values[0]))

print("write shap_values")
for i in range(len(shap_values)):
    np.savetxt("C:/venv-Shap/"+shap_reason+"/shap_"+str(i)+".csv", shap_values[i])
np.savetxt("C:/venv-Shap/"+shap_reason+"/shape.csv",np.array([len(shap_values)]))

column_names = X.columns.values
print(column_names)
pd.DataFrame(column_names, columns=['Column Names']).to_csv("C:/venv-Shap/"+shap_reason+'/columns.csv', index=False)
pd.DataFrame(le.classes_, columns=['Class Names']).to_csv(f"C:/venv-Shap/{shap_reason}/class_names.csv", index=False)
exit()

# print("Writing SHAP values...")
# # shap_values is a list of arrays: one for each class
# for i, class_shap in enumerate(shap_values):
#     np.savetxt(f"./{shap_reason}/shap_class_{i}.csv", class_shap)
# np.savetxt(f"./{shap_reason}/shape.csv", np.array([len(shap_values)]))
# # Save column names
# column_names = X.columns.values
# pd.DataFrame(column_names, columns=['Column Names']).to_csv(f"./{shap_reason}/columns.csv", index=False)
# print("SHAP values saved successfully!")
# # ---------------------- Optional: Save SHAP base values ----------------------
# base_values = explainer.expected_value
# np.savetxt(f"./{shap_reason}/base_values.csv", np.array(base_values))
# exit()

