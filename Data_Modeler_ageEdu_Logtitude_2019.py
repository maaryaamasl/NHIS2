import time
# import pkg_resources
import sys
print("RUNNING PYTHON:", sys.executable)
import pandas
print("PANDAS VERSION:", pandas.__version__)
import numpy as np
import pandas as pd
import sklearn as sk
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support
)
# import pycaret as pc
# import tpot
# from tpot import TPOTClassifier
import h2o
from h2o.automl import H2OAutoML
# import autokeras as ak
# from autokeras import StructuredDataClassifier
import shap
# import shapley
from xgboost import XGBClassifier
# from lightgbm import LGBMClassifier
import torch
print(torch.version.cuda)
print(torch.cuda.is_available())
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import label_binarize

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
    # if filtering in column:
    print(column, set(Merged_data[column]))
# exit()


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
X = Merged_data.drop(columns=[col for col in Merged_data.columns if col.startswith('change_') or col in outcome]) ################################################################## only 2019 droped changed

Y = Merged_data[outcome]  # Target
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=42)

categorical_non_numeric = ['REGION', 'AGEP_A', 'ORIENT_A', 'MARITAL_A', 'RACEALLP_A', 'EDUC_A', 'MAXEDUC_A',
                           'change_MAXEDUC_A', 'change_EDUC_A', 'change_AGEP_A', 'change_MARITAL_A',
                           ]
##################################################################################################3
##################################################################################################3
##################################################################################################3

model = "h2" # sk, xg, h2
include_algorithms = ["DeepLearning"] # ["GBM", "DRF", "DeepLearning", "GLM"]
max_models = 20
print ("########################## SK Learn base #################################################")
if model=="sk":
    # =========================================================
    # MODELS TO COMPARE
    # =========================================================
    models = {
        'Logistic Regression': LogisticRegression(max_iter=500),
        'Random Forest': RandomForestClassifier(),
        'XGBoost': XGBClassifier(eval_metric='mlogloss'),
        # 'SVM': SVC(probability=True)
    }

    # =========================================================
    # LABEL ENCODING
    # =========================================================
    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train)
    y_test_encoded = le.transform(y_test)
    class_names = le.classes_
    n_classes = len(class_names)
    class_ids = np.arange(n_classes)

    y_test_bin = label_binarize(y_test_encoded, classes=class_ids)

    # =========================================================
    # FOR STORING FINAL HUMAN-READABLE TABLES
    # =========================================================
    all_tables = []

    # =========================================================
    # EVALUATE EACH MODEL
    # =========================================================
    for name, model in models.items():

        print(f"\n================== TRAINING {name} ==================\n")

        # ---- Train ----
        model.fit(x_train, y_train_encoded)

        # ---- Predictions ----
        y_pred = model.predict(x_test)

        # ---- Probabilities ----
        try:
            y_proba = model.predict_proba(x_test)
        except:
            y_proba = None

        # ---- GLOBAL METRICS ----
        acc = accuracy_score(y_test_encoded, y_pred)
        macro_acc = balanced_accuracy_score(y_test_encoded, y_pred)
        if y_proba is not None:
            macro_auc = roc_auc_score(y_test_bin, y_proba, average="macro", multi_class="ovr")
        else:
            macro_auc = np.nan

        # ---- PER-CLASS PREC / REC / F1 ----
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test_encoded, y_pred, labels=class_ids, zero_division=0
        )

        # ---- PER-CLASS AUC ----
        per_class_auc = {}
        if y_proba is not None:
            for i, cname in enumerate(class_names):
                try:
                    auc_i = roc_auc_score((y_test_encoded == i).astype(int), y_proba[:, i])
                except:
                    auc_i = np.nan
                per_class_auc[cname] = auc_i
        else:
            per_class_auc = {c: np.nan for c in class_names}

        # ---- SPECIFICITY ----
        cm = confusion_matrix(y_test_encoded, y_pred, labels=class_ids)
        total = cm.sum()

        specificity = []
        for i in class_ids:
            TP = cm[i, i]
            FN = cm[i, :].sum() - TP
            FP = cm[:, i].sum() - TP
            TN = total - (TP + FN + FP)
            spec_i = TN / (TN + FP) if (TN + FP) > 0 else np.nan
            specificity.append(spec_i)


        # =========================================================
        # FORMATTERS
        # =========================================================
        def fmt_per_class(values, decimals=4):
            return (
                f"inc:{values[0]:.{decimals}f}, "
                f"per:{values[1]:.{decimals}f}, "
                f"rec:{values[2]:.{decimals}f}, "
                f"res:{values[3]:.{decimals}f}"
            )


        def fmt_per_class_auc(decimals=2):
            return (
                f"inc:{per_class_auc[class_names[0]]:.{decimals}f}, "
                f"per:{per_class_auc[class_names[1]]:.{decimals}f}, "
                f"rec:{per_class_auc[class_names[2]]:.{decimals}f}, "
                f"res:{per_class_auc[class_names[3]]:.{decimals}f}"
            )


        # =========================================================
        # BUILD EXPLANATION-STYLE TABLE FOR THIS MODEL
        # =========================================================
        rows = [
            ["Accuracy",
             "Percentage of all predictions correct.",
             "Overall correctness.",
             "Misleading with imbalance.",
             "≥ 0.70 OK",
             f"{acc:.4f}",
             "—"],

            ["Macro Accuracy",
             "Average recall across classes.",
             "Balanced correctness.",
             "Fairness for minority.",
             "≥ 0.50 OK",
             f"{macro_acc:.4f}",
             fmt_per_class(recall)],

            ["Recall (Sensitivity)",
             "Found actual positives.",
             "TP / (TP + FN)",
             "Important for incidence/persistence/recovery.",
             "≥ 0.40 acceptable",
             f"{np.mean(recall):.4f}",
             fmt_per_class(recall)],

            ["Precision (PPV)",
             "Correct positive predictions.",
             "TP / (TP + FP)",
             "Avoids false alarms.",
             "≥ 0.40 acceptable",
             f"{np.mean(precision):.4f}",
             fmt_per_class(precision)],

            ["Specificity",
             "Correct identification of negatives.",
             "TN / (TN + FP)",
             "Prevents misclassification into minority classes.",
             "≥ 0.80 good",
             f"{np.mean(specificity):.4f}",
             fmt_per_class(specificity)],

            ["F1-score",
             "Balance of precision + recall.",
             "Harmonic mean.",
             "Best metric for imbalance.",
             "≥ 0.40 acceptable",
             f"{np.mean(f1):.4f}",
             fmt_per_class(f1)],

            ["Macro-AUC",
             "Average AUC across classes.",
             "Ranking ability.",
             "Most robust metric.",
             "≥ 0.70 publishable",
             f"{macro_auc:.4f}",
             fmt_per_class_auc()],
        ]

        df_table = pd.DataFrame(rows, columns=[
            "Metric", "Meaning", "What It Measures", "Why It Matters",
            "Good Range", "Macro Result", "Per-Class Results"
        ])

        print(f"\n=== {name}: EXPLANATION METRIC TABLE ===\n")
        print(df_table.to_string(index=False))

        all_tables.append((name, df_table))

# print ("########################## unsuccessful #################################################")
if False:

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

print("########################## XGboost base #######################################################") ############################### <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< Chosen model
# XG reweight + threashoulding
if model=="xg":
    from xgboost import XGBClassifier
    from sklearn.metrics import (
        accuracy_score, roc_auc_score, confusion_matrix, classification_report
    )
    from sklearn.preprocessing import LabelEncoder
    import numpy as np
    import pandas as pd


    # -------------------------------------------------
    # 0) Ensure y_train / y_test are 1D arrays
    # -------------------------------------------------
    def to_1d(y):
        if isinstance(y, pd.DataFrame):
            return y.iloc[:, 0].values
        elif isinstance(y, pd.Series):
            return y.values
        else:
            return np.asarray(y)


    y_train_arr = to_1d(y_train)
    y_test_arr = to_1d(y_test)

    # -------------------------------------------------
    # 1) Encode string labels -> integers (0..3)
    # -------------------------------------------------
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train_arr)
    y_test_enc = le.transform(y_test_arr)

    class_names = le.classes_  # ['incidence','persistence','recovery','resilience']
    n_classes = len(class_names)
    classes = np.arange(n_classes)  # [0,1,2,3]

    print("Encoded classes:", classes)
    print("Class names:", class_names)

    # -------------------------------------------------
    # 2) Compute class weights from TRAIN ONLY
    # -------------------------------------------------
    class_counts = np.bincount(y_train_enc)
    total_train = len(y_train_enc)

    class_weights = {
        cls: total_train / (n_classes * count)
        for cls, count in zip(classes, class_counts)
    }

    print("Class counts (train):", dict(zip(class_names, class_counts)))
    print("Class weights:", dict(zip(class_names, [class_weights[c] for c in classes])))

    sample_weight_train = np.array([class_weights[y] for y in y_train_enc])

    # -------------------------------------------------
    # 3) Train XGBoost with class weights
    # -------------------------------------------------
    clf = XGBClassifier(
        objective="multi:softprob",
        num_class=n_classes,
        eval_metric="mlogloss",
        random_state=1,
    )

    clf.fit(
        x_train,
        y_train_enc,
        sample_weight=sample_weight_train
    )

    # -------------------------------------------------
    # 4) Predict on TEST set
    # -------------------------------------------------
    y_pred_enc = clf.predict(x_test)
    y_pred_proba = clf.predict_proba(x_test)

    # -------------------------------------------------
    # PER-CLASS AUC (One-vs-Rest)
    # -------------------------------------------------
    per_class_auc = {}

    # One-hot encode y_test
    y_test_bin = np.eye(n_classes)[y_test_enc]  # shape (N, 4)

    for i, cname in enumerate(class_names):
        # true binary labels for class i vs rest
        y_true_i = y_test_bin[:, i]

        # predicted probabilities for class i
        y_proba_i = y_pred_proba[:, i]

        try:
            auc_i = roc_auc_score(y_true_i, y_proba_i)
        except ValueError:
            auc_i = np.nan  # if a class has no positives in test set

        per_class_auc[cname] = auc_i

    print("\n=== PER-CLASS AUC (OVR) ===")
    for cname in class_names:
        print(f"{cname}: {per_class_auc[cname]:.3f}")

    # -------------------------------------------------
    # 5) Accuracy & macro AUC (ovr)
    # -------------------------------------------------
    accuracy = accuracy_score(y_test_enc, y_pred_enc)

    # one-hot encode test labels to match probability columns
    y_test_bin = np.eye(n_classes)[y_test_enc]

    macro_auc = roc_auc_score(
        y_test_bin,
        y_pred_proba,
        multi_class="ovr",
        average="macro"
    )

    print("\n=== GLOBAL METRICS (TEST) ===")
    print("Accuracy:", accuracy)
    print("Macro AUC:", macro_auc)

    # -------------------------------------------------
    # 6) Confusion matrix
    # -------------------------------------------------
    cm = confusion_matrix(y_test_enc, y_pred_enc, labels=classes)

    cm_df = pd.DataFrame(
        cm,
        index=[f"true_{name}" for name in class_names],
        columns=[f"pred_{name}" for name in class_names]
    )

    print("\n=== CONFUSION MATRIX (TEST) ===")
    print(cm_df)

    # -------------------------------------------------
    # 7) Class-specific Sensitivity / Specificity / PPV / F1
    # -------------------------------------------------
    metrics = []
    total = cm.sum()

    for i, cls in enumerate(classes):
        TP = cm[i, i]
        FN = cm[i, :].sum() - TP
        FP = cm[:, i].sum() - TP
        TN = total - (TP + FN + FP)

        sensitivity = TP / (TP + FN) if (TP + FN > 0) else np.nan
        specificity = TN / (TN + FP) if (TN + FP > 0) else np.nan
        ppv = TP / (TP + FP) if (TP + FP > 0) else np.nan
        f1 = 2 * ppv * sensitivity / (ppv + sensitivity) if (ppv + sensitivity) else np.nan

        metrics.append({
            "class_index": cls,
            "class_name": class_names[i],
            "TP": TP, "FP": FP, "FN": FN, "TN": TN,
            "Sensitivity": sensitivity,
            "Specificity": specificity,
            "PPV": ppv,
            "F1": f1
        })

    metrics_df = pd.DataFrame(metrics)

    print("\n=== CLASS-SPECIFIC METRICS (TEST) ===")
    print(metrics_df)

    # -------------------------------------------------
    # 8) Macro-averaged metrics (for manuscript)
    # -------------------------------------------------
    macro_sensitivity = metrics_df["Sensitivity"].mean()
    macro_specificity = metrics_df["Specificity"].mean()
    macro_ppv = metrics_df["PPV"].mean()
    macro_f1 = metrics_df["F1"].mean()

    print("\n=== MACRO-AVERAGED METRICS (TEST) ===")
    print(f"Macro Sensitivity: {macro_sensitivity:.3f}")
    print(f"Macro Specificity: {macro_specificity:.3f}")
    print(f"Macro PPV:         {macro_ppv:.3f}")
    print(f"Macro F1:          {macro_f1:.3f}")

    # -------------------------------------------------
    # 9) Sklearn classification report (just for reference)
    # -------------------------------------------------
    print("\n=== SKLEARN CLASSIFICATION REPORT (TEST) ===")
    print(classification_report(y_test_enc, y_pred_enc, target_names=class_names))

    print("\n=== POST-HOC ADJUSTMENT FOR IMBALANCE (REWEIGHTED PROBAS) ===")

    # 1) Start from the class_weights we already computed
    #    (they were inverse-frequency; quite strong)
    #    We'll use a softer version so we don't completely destroy specificity.
    weight_vec = np.array([class_weights[c] for c in classes], dtype=float)

    # Option: soften them (e.g., square root) so effect is not too extreme
    weight_vec_soft = np.sqrt(weight_vec)

    print("Original class weights:     ", weight_vec)
    print("Soft reweighting (sqrt):   ", weight_vec_soft)

    # 2) Reweight predicted probabilities
    # shape: (n_samples, n_classes)
    proba_adj = y_pred_proba * weight_vec_soft  # broadcast along columns
    proba_adj = proba_adj / proba_adj.sum(axis=1, keepdims=True)  # renormalize

    # 3) New predicted labels after reweighting
    y_pred_adj = proba_adj.argmax(axis=1)

    # 4) New confusion matrix
    cm_adj = confusion_matrix(y_test_enc, y_pred_adj, labels=classes)
    cm_adj_df = pd.DataFrame(
        cm_adj,
        index=[f"true_{name}" for name in class_names],
        columns=[f"pred_{name}" for name in class_names],
    )

    print("\nConfusion Matrix (reweighted proba):")
    print(cm_adj_df)

    # 5) Recompute class-specific metrics for adjusted predictions
    metrics_adj = []
    total_adj = cm_adj.sum()

    for i, cls in enumerate(classes):
        TP = cm_adj[i, i]
        FN = cm_adj[i, :].sum() - TP
        FP = cm_adj[:, i].sum() - TP
        TN = total_adj - (TP + FN + FP)

        sensitivity = TP / (TP + FN) if (TP + FN > 0) else np.nan
        specificity = TN / (TN + FP) if (TN + FP > 0) else np.nan
        ppv = TP / (TP + FP) if (TP + FP > 0) else np.nan
        f1 = 2 * ppv * sensitivity / (ppv + sensitivity) if (ppv + sensitivity) else np.nan

        metrics_adj.append({
            "class_index": cls,
            "class_name": class_names[i],
            "TP": TP, "FP": FP, "FN": FN, "TN": TN,
            "Sensitivity": sensitivity,
            "Specificity": specificity,
            "PPV": ppv,
            "F1": f1
        })

    metrics_adj_df = pd.DataFrame(metrics_adj)
    print("\nClass-specific metrics (reweighted proba):")
    print(metrics_adj_df)

    # 6) Macro-averaged metrics for adjusted predictions
    macro_sens_adj = metrics_adj_df["Sensitivity"].mean()
    macro_spec_adj = metrics_adj_df["Specificity"].mean()
    macro_ppv_adj = metrics_adj_df["PPV"].mean()
    macro_f1_adj = metrics_adj_df["F1"].mean()
    acc_adj = accuracy_score(y_test_enc, y_pred_adj)

    print("\n=== MACRO METRICS (REWEIGHTED PROBA) ===")
    print(f"Accuracy (adj):          {acc_adj:.3f}")
    print(f"Macro Sensitivity (adj): {macro_sens_adj:.3f}")
    print(f"Macro Specificity (adj): {macro_spec_adj:.3f}")
    print(f"Macro PPV (adj):         {macro_ppv_adj:.3f}")
    print(f"Macro F1 (adj):          {macro_f1_adj:.3f}")

    your_metrics = {
        "Accuracy": acc_adj,  # or accuracy
        "Recall (Macro)": macro_sens_adj,  # sensitivity
        "Precision (Macro)": macro_ppv_adj,  # PPV
        "Specificity (Macro)": macro_spec_adj,  # TNR
        "F1-score (Macro)": macro_f1_adj,  # F1
        "Macro-AUC": macro_auc,  # AUC
        "Macro-F1": macro_f1_adj  # same as F1 macro
    }

    import pandas as pd

    # === FILL from your computed results ===
    your_metrics = {
        "Accuracy": acc_adj,
        "Recall (Macro)": macro_sens_adj,
        "Precision (Macro)": macro_ppv_adj,
        "Specificity (Macro)": macro_spec_adj,
        "F1-score (Macro)": macro_f1_adj,
        "Macro-AUC": macro_auc,  # overall macro AUC
        "Macro-F1": macro_f1_adj
    }

    # metrics_adj_df must contain:
    # class_name | Sensitivity | Specificity | PPV | F1
    class_metrics = metrics_adj_df.set_index("class_name")


    def format_per_class(col):
        """Return compact per-class string: inc/pers/rec/res for a metric column."""
        return (
            f"inc:{class_metrics[col].loc['incidence']:.2f}, "
            f"per:{class_metrics[col].loc['persistence']:.2f}, "
            f"rec:{class_metrics[col].loc['recovery']:.2f}, "
            f"res:{class_metrics[col].loc['resilience']:.2f}"
        )


    def format_per_class_auc():
        """Return compact per-class AUC string using per_class_auc dict."""
        return (
            f"inc:{per_class_auc['incidence']:.2f}, "
            f"per:{per_class_auc['persistence']:.2f}, "
            f"rec:{per_class_auc['recovery']:.2f}, "
            f"res:{per_class_auc['resilience']:.2f}"
        )


    rows = [
        ["Accuracy",
         "Percentage of all predictions that were correct.",
         "Overall correctness.",
         "Misleading with imbalance.",
         "≥ 0.70 (but limited meaning)",
         f"{your_metrics['Accuracy']:.3f}",
         "—"],

        ["Recall (Sensitivity)",
         "Of all actual cases, how many were detected?",
         "TP / (TP + FN)",
         "Critical for minority classes.",
         "≥ 0.40 acceptable",
         f"{your_metrics['Recall (Macro)']:.3f}",
         format_per_class("Sensitivity")],

        ["Precision (PPV)",
         "Of predicted positives, how many were correct?",
         "TP / (TP + FP)",
         "Avoids false alarms.",
         "≥ 0.40 acceptable",
         f"{your_metrics['Precision (Macro)']:.3f}",
         format_per_class("PPV")],

        ["Specificity",
         "Correctly identifies negatives.",
         "TN / (TN + FP)",
         "Prevents mislabeling into minority classes.",
         "≥ 0.80 good",
         f"{your_metrics['Specificity (Macro)']:.3f}",
         format_per_class("Specificity")],

        ["F1-score",
         "Harmonic mean of precision + recall.",
         "Balance of miss vs false alarm.",
         "Best single metric for imbalance.",
         "≥ 0.40 acceptable",
         f"{your_metrics['F1-score (Macro)']:.3f}",
         format_per_class("F1")],

        ["Macro-AUC",
         "Averaged AUC across classes.",
         "Ranking ability.",
         "Most robust for imbalance.",
         "≥ 0.70 publishable",
         f"{your_metrics['Macro-AUC']:.3f}",
         format_per_class_auc()],  # <-- per-class AUC here

        ["Macro-F1",
         "F1 averaged equally across classes.",
         "How well each class is predicted.",
         "Best fairness metric.",
         "≥ 0.45 acceptable",
         f"{your_metrics['Macro-F1']:.3f}",
         format_per_class("F1")]
    ]

    df = pd.DataFrame(rows, columns=[
        "Metric",
        "Meaning",
        "What It Measures",
        "Why It Matters",
        "Good Range",
        "Your Macro Result",
        "Per-Class Results"
    ])

    print("\n=== MODEL METRIC EXPLANATION TABLE (WITH PER-CLASS RESULTS & AUC) ===\n")
    print(df.to_string(index=False))

# print ("########################## unsuccessful #################################################")
if False:
    if False:
        # ============================
        # 0) Imports
        # ============================
        import numpy as np
        import pandas as pd

        from xgboost import XGBClassifier
        from sklearn.preprocessing import LabelEncoder
        from sklearn.metrics import (
            accuracy_score, f1_score, confusion_matrix, classification_report
        )

        from imblearn.over_sampling import RandomOverSampler  # pip install imbalanced-learn


        # ============================
        # 1) Helper: make y 1D
        # ============================
        def to_1d(y):
            if isinstance(y, pd.DataFrame):
                return y.iloc[:, 0].values
            elif isinstance(y, pd.Series):
                return y.values
            else:
                return np.asarray(y)


        y_train_arr = to_1d(y_train)
        y_test_arr = to_1d(y_test)

        # ============================
        # 2) Encode labels -> integers (0..3)
        # ============================
        le = LabelEncoder()
        y_train_enc = le.fit_transform(y_train_arr)
        y_test_enc = le.transform(y_test_arr)

        class_names = le.classes_  # e.g. ['incidence','persistence','recovery','resilience']
        n_classes = len(class_names)
        classes = np.arange(n_classes)

        print("Classes:", class_names)

        # ============================
        # 3) Random Oversampling on TRAIN ONLY
        #    (oversample minority classes to match majority)
        # ============================
        ros = RandomOverSampler(random_state=1)
        X_train_res, y_train_res = ros.fit_resample(x_train, y_train_enc)

        print("Original train counts:", dict(zip(*np.unique(y_train_enc, return_counts=True))))
        print("Resampled train counts:", dict(zip(*np.unique(y_train_res, return_counts=True))))

        # ============================
        # 4) Train XGBoost on oversampled data
        # ============================
        clf = XGBClassifier(
            objective="multi:softprob",
            num_class=n_classes,
            eval_metric="mlogloss",
            random_state=1,
            tree_method="hist",
            max_depth=6,
            n_estimators=400,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
        )

        clf.fit(X_train_res, y_train_res)

        # ============================
        # 5) Baseline predictions on TEST (no threshold tuning)
        # ============================
        proba_test = clf.predict_proba(x_test)  # shape (N_test, n_classes)
        y_pred_base = proba_test.argmax(axis=1)

        acc_base = accuracy_score(y_test_enc, y_pred_base)
        macro_f1_base = f1_score(y_test_enc, y_pred_base, average="macro")

        print("\n=== BASELINE (oversampled, argmax) ===")
        print("Accuracy:", acc_base)
        print("Macro F1:", macro_f1_base)

        cm_base = confusion_matrix(y_test_enc, y_pred_base, labels=classes)
        print("\nConfusion matrix (baseline):")
        print(pd.DataFrame(
            cm_base,
            index=[f"true_{c}" for c in class_names],
            columns=[f"pred_{c}" for c in class_names],
        ))

        print("\nClassification report (baseline):")
        print(classification_report(y_test_enc, y_pred_base, target_names=class_names))

        # ============================
        # 6) Threshold / weight tuning on probabilities
        #    Here: down-weight the majority class ('resilience')
        #         and pick the best weight for Macro F1.
        # ============================

        # Find index of 'resilience' (majority class) if present
        try:
            idx_res = list(class_names).index("resilience")
        except ValueError:
            # If different name, adjust here
            idx_res = np.argmax(np.bincount(y_train_enc))  # fallback: largest class

        res_weight_grid = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4]  # shrink majority prob

        best_macro_f1 = -1.0
        best_weight = 1.0
        best_pred_adj = None

        for w in res_weight_grid:
            # start with all ones
            class_weights = np.ones(n_classes)
            class_weights[idx_res] = w  # only shrink resilience

            # reweight probabilities
            proba_adj = proba_test * class_weights[None, :]
            proba_adj = proba_adj / proba_adj.sum(axis=1, keepdims=True)

            y_pred_adj = proba_adj.argmax(axis=1)

            macro_f1 = f1_score(y_test_enc, y_pred_adj, average="macro")

            print(f"Resilience weight {w:.2f} -> Macro F1 = {macro_f1:.3f}")

            if macro_f1 > best_macro_f1:
                best_macro_f1 = macro_f1
                best_weight = w
                best_pred_adj = y_pred_adj.copy()

        # ============================
        # 7) Evaluate best threshold-tuned model
        # ============================
        print("\n=== BEST THRESHOLD-TUNED MODEL ===")
        print(f"Best resilience weight: {best_weight:.2f}")
        print(f"Best Macro F1: {best_macro_f1:.3f}")
        acc_best = accuracy_score(y_test_enc, best_pred_adj)
        print(f"Accuracy: {acc_best:.3f}")

        cm_best = confusion_matrix(y_test_enc, best_pred_adj, labels=classes)
        cm_best_df = pd.DataFrame(
            cm_best,
            index=[f"true_{c}" for c in class_names],
            columns=[f"pred_{c}" for c in class_names],
        )
        print("\nConfusion matrix (best tuned):")
        print(cm_best_df)

        print("\nClass-specific metrics (best tuned):")
        # compute per-class precision/recall/F1 like before
        for i, cname in enumerate(class_names):
            TP = cm_best[i, i]
            FN = cm_best[i, :].sum() - TP
            FP = cm_best[:, i].sum() - TP
            TN = cm_best.sum() - (TP + FN + FP)

            sens = TP / (TP + FN) if (TP + FN) > 0 else np.nan
            spec = TN / (TN + FP) if (TN + FP) > 0 else np.nan
            ppv = TP / (TP + FP) if (TP + FP) > 0 else np.nan
            f1 = 2 * ppv * sens / (ppv + sens) if (ppv + sens) > 0 else np.nan

            print(
                f"{cname:12s} | Sensitivity={sens:.3f}  Specificity={spec:.3f}  "
                f"Precision={ppv:.3f}  F1={f1:.3f}"
            )

        print("\nClassification report (best tuned):")
        print(classification_report(y_test_enc, best_pred_adj, target_names=class_names))

    if False:
        import numpy as np
        import pandas as pd
        import xgboost as xgb
        from scipy.special import softmax
        from sklearn.preprocessing import LabelEncoder
        from sklearn.metrics import (
            accuracy_score, roc_auc_score, confusion_matrix, classification_report
        )


        # ============================================================
        # 0) Encode labels
        # ============================================================
        def to_1d(y):
            if isinstance(y, pd.DataFrame):
                return y.iloc[:, 0].values
            elif isinstance(y, pd.Series):
                return y.values
            else:
                return np.asarray(y)


        y_train_arr = to_1d(y_train)
        y_test_arr = to_1d(y_test)

        le = LabelEncoder()
        y_train_enc = le.fit_transform(y_train_arr)
        y_test_enc = le.transform(y_test_arr)

        class_names = le.classes_
        n_classes = len(class_names)
        classes = np.arange(n_classes)

        print("Classes:", class_names)


        # ============================================================
        # 1) Balanced Softmax Loss
        # ============================================================
        def balanced_softmax_loss(preds, dtrain, num_class, class_freq):
            """
            Balanced Softmax (Ren et al. 2020) for extreme imbalance.
            Uses priors log(freq).
            """
            # reshape logits
            logits = preds.reshape(-1, num_class)

            # subtract log frequencies (like class priors)
            prior_corrected = logits - np.log(class_freq + 1e-9)

            # softmax
            prob = softmax(prior_corrected, axis=1)

            # true labels
            labels = dtrain.get_label().astype(int)

            # one-hot
            onehot = np.eye(num_class)[labels]

            # gradient & hessian (cross-entropy style)
            grad = (prob - onehot)
            hess = prob * (1 - prob)

            return grad.ravel(), hess.ravel()


        # ============================================================
        # 2) Class frequencies (priors)
        # ============================================================
        class_counts = np.bincount(y_train_enc)
        class_freq = class_counts / class_counts.sum()

        print("Class counts:", dict(zip(class_names, class_counts)))
        print("Class frequencies:", dict(zip(class_names, class_freq)))

        # ============================================================
        # 3) Build DMatrices
        # ============================================================
        dtrain = xgb.DMatrix(x_train, label=y_train_enc)
        dtest = xgb.DMatrix(x_test, label=y_test_enc)

        base_params = {
            "num_class": n_classes,
            "eta": 0.03,
            "max_depth": 6,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "min_child_weight": 3,
            "tree_method": "hist",
        }

        num_round = 400

        # Train model
        bst = xgb.train(
            base_params,
            dtrain,
            num_boost_round=num_round,
            obj=lambda preds, d: balanced_softmax_loss(
                preds, d, num_class=n_classes, class_freq=class_freq
            ),
        )

        # ============================================================
        # 4) Predict on test
        # ============================================================
        logits_test = bst.predict(dtest, output_margin=True)
        proba_test = softmax(logits_test - np.log(class_freq + 1e-9), axis=1)
        y_pred_enc = proba_test.argmax(axis=1)

        # ============================================================
        # 5) Global metrics
        # ============================================================
        accuracy = accuracy_score(y_test_enc, y_pred_enc)

        y_test_bin = np.eye(n_classes)[y_test_enc]
        macro_auc = roc_auc_score(
            y_test_bin,
            proba_test,
            multi_class="ovr",
            average="macro"
        )

        print("\n=== BALANCED SOFTMAX: GLOBAL METRICS (TEST) ===")
        print("Accuracy:", accuracy)
        print("Macro AUC:", macro_auc)

        # ============================================================
        # 6) Confusion matrix
        # ============================================================
        cm = confusion_matrix(y_test_enc, y_pred_enc, labels=classes)
        print("\nConfusion Matrix:")
        print(pd.DataFrame(
            cm,
            index=[f"true_{c}" for c in class_names],
            columns=[f"pred_{c}" for c in class_names]
        ))

        # ============================================================
        # 7) Class-specific metrics
        # ============================================================
        metrics = []
        total = cm.sum()

        for i, cls in enumerate(classes):
            TP = cm[i, i]
            FN = cm[i, :].sum() - TP
            FP = cm[:, i].sum() - TP
            TN = total - (TP + FN + FP)

            sens = TP / (TP + FN) if TP + FN > 0 else np.nan
            spec = TN / (TN + FP) if TN + FP > 0 else np.nan
            ppv = TP / (TP + FP) if TP + FP > 0 else np.nan
            f1 = 2 * ppv * sens / (ppv + sens) if (ppv + sens) else np.nan

            metrics.append({
                "class": class_names[i],
                "Sensitivity": sens,
                "Specificity": spec,
                "PPV": ppv,
                "F1": f1
            })

        metrics_df = pd.DataFrame(metrics)
        print("\nClass-specific metrics:")
        print(metrics_df)

        # ============================================================
        # 8) Macro metrics for manuscript
        # ============================================================
        print("\n=== MACRO-AVERAGED METRICS (Balanced Softmax) ===")
        print("Macro Sensitivity:", metrics_df["Sensitivity"].mean())
        print("Macro Specificity:", metrics_df["Specificity"].mean())
        print("Macro PPV:", metrics_df["PPV"].mean())
        print("Macro F1:", metrics_df["F1"].mean())

        print("\n=== Classification Report ===")
        print(classification_report(y_test_enc, y_pred_enc, target_names=class_names))

    if False:
        import numpy as np
        import pandas as pd
        import xgboost as xgb
        from scipy.special import softmax
        from sklearn.preprocessing import LabelEncoder
        from sklearn.metrics import (
            accuracy_score, roc_auc_score, confusion_matrix, classification_report
        )


        # ============================================================
        # 0) Prep labels: string -> integer (0..3)
        # ============================================================
        def to_1d(y):
            if isinstance(y, pd.DataFrame):
                return y.iloc[:, 0].values
            elif isinstance(y, pd.Series):
                return y.values
            else:
                return np.asarray(y)


        y_train_arr = to_1d(y_train)
        y_test_arr = to_1d(y_test)

        le = LabelEncoder()
        y_train_enc = le.fit_transform(y_train_arr)
        y_test_enc = le.transform(y_test_arr)

        class_names = le.classes_
        n_classes = len(class_names)
        classes = np.arange(n_classes)

        print("Encoded classes:", classes)
        print("Class names:", class_names)


        # ============================================================
        # 1) Multiclass focal loss objective (generic)
        # ============================================================
        def focal_multiclass(preds: np.ndarray,
                             dtrain: xgb.DMatrix,
                             num_class: int,
                             gamma: float = 2.0,
                             alpha: float = 0.25):
            """
            Multiclass focal loss for XGBoost.
            preds: flat array of shape (N * num_class,) of raw logits.
            Returns grad, hess as flat arrays.
            """
            # reshape to (N, C)
            preds = preds.reshape(-1, num_class)

            # softmax to probs
            prob = softmax(preds, axis=1)  # (N, C)

            # one-hot labels
            y = dtrain.get_label().astype(int)
            onehot = np.eye(num_class)[y]  # (N, C)

            # p_t: prob of true class
            pt = (onehot * prob).sum(axis=1, keepdims=True)  # (N, 1)
            pt_clipped = np.clip(pt, 1e-7, 1.0)

            # grad
            grad = (alpha * (1.0 - pt) ** gamma) * (prob - onehot)

            # hess (approx, but works in practice)
            focal_factor = alpha * (1.0 - pt) ** gamma * (
                    1.0 + gamma * pt * (-np.log(pt_clipped))
            )
            hess = focal_factor * prob * (1.0 - prob)

            return grad.ravel(), hess.ravel()


        # ============================================================
        # 2) Build DMatrices
        # ============================================================
        dtrain = xgb.DMatrix(x_train, label=y_train_enc)
        dtest = xgb.DMatrix(x_test, label=y_test_enc)

        base_params = {
            "num_class": n_classes,
            "eta": 0.05,
            "max_depth": 6,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "min_child_weight": 3,
            "tree_method": "hist",
            "eval_metric": "mlogloss",  # just for logging
        }

        num_round = 400
        gamma_list = [0.25, 0.5, 1.0, 2.0]

        summary_rows = []

        for gamma in gamma_list:
            print(f"\n==============================")
            print(f"Training focal XGBoost with gamma = {gamma}")
            print(f"==============================")

            # Train with custom objective for this gamma
            bst = xgb.train(
                base_params,
                dtrain,
                num_boost_round=num_round,
                obj=lambda preds, d, g=gamma: focal_multiclass(
                    preds, d, num_class=n_classes, gamma=g, alpha=0.25
                ),
                verbose_eval=False,
            )

            # ---- Predictions ----
            logits_test = bst.predict(dtest, output_margin=True)  # (N, C)
            y_pred_proba = softmax(logits_test, axis=1)
            y_pred_enc = y_pred_proba.argmax(axis=1)

            # ---- Global metrics ----
            accuracy = accuracy_score(y_test_enc, y_pred_enc)
            y_test_bin = np.eye(n_classes)[y_test_enc]
            macro_auc = roc_auc_score(
                y_test_bin,
                y_pred_proba,
                multi_class="ovr",
                average="macro"
            )

            # ---- Confusion matrix ----
            cm = confusion_matrix(y_test_enc, y_pred_enc, labels=classes)
            cm_df = pd.DataFrame(
                cm,
                index=[f"true_{name}" for name in class_names],
                columns=[f"pred_{name}" for name in class_names],
            )

            print("\nConfusion Matrix (test):")
            print(cm_df)

            # ---- Class-specific metrics ----
            metrics = []
            total = cm.sum()

            for i, cls in enumerate(classes):
                TP = cm[i, i]
                FN = cm[i, :].sum() - TP
                FP = cm[:, i].sum() - TP
                TN = total - (TP + FN + FP)

                sensitivity = TP / (TP + FN) if (TP + FN > 0) else np.nan
                specificity = TN / (TN + FP) if (TN + FP > 0) else np.nan
                ppv = TP / (TP + FP) if (TP + FP > 0) else np.nan
                f1 = 2 * ppv * sensitivity / (ppv + sensitivity) if (ppv + sensitivity) else np.nan

                metrics.append({
                    "class_index": cls,
                    "class_name": class_names[i],
                    "TP": TP, "FP": FP, "FN": FN, "TN": TN,
                    "Sensitivity": sensitivity,
                    "Specificity": specificity,
                    "PPV": ppv,
                    "F1": f1
                })

            metrics_df = pd.DataFrame(metrics)
            print("\nClass-specific metrics (test, focal loss, gamma={}):".format(gamma))
            print(metrics_df)

            # ---- Macro-averaged metrics ----
            macro_sensitivity = metrics_df["Sensitivity"].mean()
            macro_specificity = metrics_df["Specificity"].mean()
            macro_ppv = metrics_df["PPV"].mean()
            macro_f1 = metrics_df["F1"].mean()

            print("\n=== MACRO-AVERAGED METRICS (TEST, gamma={}) ===".format(gamma))
            print(f"Accuracy:          {accuracy:.3f}")
            print(f"Macro AUC:         {macro_auc:.3f}")
            print(f"Macro Sensitivity: {macro_sensitivity:.3f}")
            print(f"Macro Specificity: {macro_specificity:.3f}")
            print(f"Macro PPV:         {macro_ppv:.3f}")
            print(f"Macro F1:          {macro_f1:.3f}")

            summary_rows.append({
                "gamma": gamma,
                "Accuracy": accuracy,
                "Macro_AUC": macro_auc,
                "Macro_Sensitivity": macro_sensitivity,
                "Macro_Specificity": macro_specificity,
                "Macro_PPV": macro_ppv,
                "Macro_F1": macro_f1,
                "Incidence_Recall": metrics_df.loc[metrics_df["class_name"] == class_names[0], "Sensitivity"].values[0],
                "Recovery_Recall": metrics_df.loc[metrics_df["class_name"] == class_names[2], "Sensitivity"].values[0],
            })

        # ============================================================
        # 3) Summary table across all gamma values
        # ============================================================
        summary_df = pd.DataFrame(summary_rows)
        print("\n\n================ SUMMARY OVER GAMMA VALUES ================")
        print(summary_df)

    if False:
        classes, counts = np.unique(Y_encoded, return_counts=True)
        n_classes = len(classes)
        total = len(Y_encoded)

        # Inverse-frequency class weights (balanced)
        class_weights = {cls: total / (n_classes * cnt) for cls, cnt in zip(classes, counts)}
        print("Class counts:", dict(zip(classes, counts)))
        print("Class weights:", class_weights)

        # Turn into sample-level weights
        sample_weight = np.array([class_weights[y] for y in Y_encoded])

        clf = XGBClassifier(
            objective="multi:softprob",
            num_class=n_classes,
            eval_metric="mlogloss",
            random_state=1,
        )
        clf.fit(X, Y_encoded, sample_weight=sample_weight)
        # clf = XGBClassifier()
        # clf.fit(X, Y_encoded)
        y_pred_clf = clf.predict(X)
        accuracy = accuracy_score(Y_encoded, y_pred_clf)
        y_pred_proba = clf.predict_proba(X)
        auc = roc_auc_score(Y_encoded, y_pred_proba, multi_class='ovr')
        print("Accuracy (on full data):", outcome, accuracy, auc)



        y_pred_clf = clf.predict(X)
        y_pred_proba = clf.predict_proba(X)

        accuracy = accuracy_score(Y_encoded, y_pred_clf)
        auc_macro = roc_auc_score(
            np.eye(n_classes)[Y_encoded],  # one-hot
            y_pred_proba,
            multi_class='ovr',
            average='macro'
        )
        print("Accuracy (on full data):", outcome, accuracy, auc_macro)

        # -----------------------------------------
        # 4) Confusion matrix & class-specific metrics
        # -----------------------------------------
        cm = confusion_matrix(Y_encoded, y_pred_clf, labels=classes)
        cm_df = pd.DataFrame(
            cm,
            index=[f"true_{c}" for c in classes],
            columns=[f"pred_{c}" for c in classes]
        )
        print("\nConfusion matrix:")
        print(cm_df)

        metrics = []
        total_cm = cm.sum()

        for i, cls in enumerate(classes):
            TP = cm[i, i]
            FN = cm[i, :].sum() - TP
            FP = cm[:, i].sum() - TP
            TN = total_cm - (TP + FN + FP)

            sensitivity = TP / (TP + FN) if (TP + FN) else np.nan     # recall
            specificity = TN / (TN + FP) if (TN + FP) else np.nan
            ppv = TP / (TP + FP) if (TP + FP) else np.nan             # precision / PPV
            f1 = 2 * ppv * sensitivity / (ppv + sensitivity) if (ppv + sensitivity) else np.nan

            metrics.append({
                "class": cls,
                "TP": TP,
                "FP": FP,
                "FN": FN,
                "TN": TN,
                "Sensitivity": sensitivity,
                "Specificity": specificity,
                "PPV": ppv,
                "F1": f1,
            })

        metrics_df = pd.DataFrame(metrics)
        print("\nClass-specific metrics (full data):")
        print(metrics_df)

        # -----------------------------------------
        # 5) Macro-averaged metrics (for manuscript)
        # -----------------------------------------
        macro_sensitivity = metrics_df["Sensitivity"].mean()
        macro_specificity = metrics_df["Specificity"].mean()
        macro_ppv = metrics_df["PPV"].mean()
        macro_f1 = metrics_df["F1"].mean()

        print("\nMacro-averaged metrics (full data):")
        print(f"Macro Sensitivity: {macro_sensitivity:.3f}")
        print(f"Macro Specificity: {macro_specificity:.3f}")
        print(f"Macro PPV:         {macro_ppv:.3f}")
        print(f"Macro F1:          {macro_f1:.3f}")

        print("\nSklearn classification report (precision/recall/F1):")
        print(classification_report(Y_encoded, y_pred_clf, labels=classes))

# H2o
print ("########################## H2O #################################################")
if model=="h2":
    from sklearn.utils.class_weight import compute_class_weight
    from sklearn.metrics import (
        accuracy_score, roc_auc_score, confusion_matrix, classification_report
    )
    from sklearn.preprocessing import label_binarize
    import numpy as np
    import pandas as pd
    import h2o
    from h2o.automl import H2OAutoML

    # ==========================
    # 1) Class weights in pandas
    # ==========================
    y_name = Y.columns.tolist()[0]

    classes = np.unique(Merged_data[y_name])
    class_wts = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=Merged_data[y_name]
    )
    wt_map = dict(zip(classes, class_wts))

    Merged_data["class_weight"] = Merged_data[y_name].map(wt_map)

    print("Class weights map:", wt_map)

    # ==========================
    # 2) Init H2O + prepare frame
    # ==========================
    print("h2o init...")
    h2o.init(max_mem_size="8G")

    x = X.columns.tolist()
    y = y_name

    # Drop change_ columns (like before)
    cleaned_data = Merged_data.drop(
        columns=[col for col in Merged_data.columns if col.startswith("change_")]
    )

    cleaned_data_h2o = h2o.H2OFrame(cleaned_data)
    cleaned_data_h2o[y] = cleaned_data_h2o[y].asfactor()

    print(len(x), x, "\n", y)

    # ==========================
    # 3) Train / test split
    # ==========================
    train, test = cleaned_data_h2o.split_frame(ratios=[0.8], seed=1)

    # ==========================
    # 4) AutoML with weights
    # ==========================

    aml = H2OAutoML(
        max_models=max_models,
        seed=1,
        include_algos=include_algorithms,
        sort_metric="auc"
    )
    aml.train(
        x=x,
        y=y,
        training_frame=train,
        weights_column="class_weight"
    )

    leader_model = aml.leader

    # ==========================
    # 5) Leaderboard + per-model metrics
    # ==========================
    leaderboard = aml.leaderboard
    print(leaderboard)

    leaderboard_all_metrics = leaderboard.as_data_frame()
    print(leaderboard_all_metrics)

    results = []

    model_ids = leaderboard_all_metrics["model_id"].tolist()
    for model_id in model_ids:
        try:
            model = h2o.get_model(model_id)

            # Predict on test
            pred_df = model.predict(test).as_data_frame()
            y_pred = pred_df["predict"].values
            # Class probabilities (all columns except 'predict')
            y_score = pred_df.drop(columns=["predict"]).values

            # True labels
            y_true = test[y].as_data_frame().values.flatten()

            # Accuracy
            acc = accuracy_score(y_true, y_pred)

            # AUC (macro OvR)
            y_true_bin = label_binarize(y_true, classes=sorted(cleaned_data[y].unique()))
            auc_macro = roc_auc_score(y_true_bin, y_score, average="macro", multi_class="ovr")

            print(f"{model_id} --> Accuracy: {acc:.4f}, Macro-Averaged AUC: {auc_macro:.4f}")
            results.append({"Model": model_id, "Accuracy": acc, "AUC_macro": auc_macro})

        except Exception as e:
            print(f"Could not evaluate model {model_id}: {e}")

    results_df = pd.DataFrame(results).sort_values(by="Accuracy", ascending=False)
    print("\n=== Model-level Accuracy & Macro-AUC (sklearn) ===")
    print(results_df)

    print("\nOutcome:", outcome)

    # ==========================
    # 6) Evaluate FINAL (leader) model in detail
    # ==========================
    final_model = aml.leader
    y_col = y

    # Predictions on test set
    pred_h2o = final_model.predict(test).as_data_frame()

    # True labels
    y_true = test[y_col].as_data_frame().iloc[:, 0]
    class_labels = sorted(y_true.unique())
    print("Unique classes in test set:", class_labels)

    # Predicted class labels
    y_pred = pred_h2o["predict"]

    # Class probabilities (columns for each class)
    prob_cols = [c for c in pred_h2o.columns if c != "predict"]
    y_proba = pred_h2o[prob_cols].values

    # --- Per-class AUC (OvR) for final model ---
    y_true_bin = label_binarize(y_true, classes=class_labels)
    per_class_auc = {}
    for i, cls in enumerate(class_labels):
        try:
            auc_i = roc_auc_score(y_true_bin[:, i], y_proba[:, i])
        except ValueError:
            auc_i = np.nan
        per_class_auc[cls] = auc_i

    macro_auc_final = np.nanmean(list(per_class_auc.values()))
    print("\n=== PER-CLASS AUC (OVR, final model) ===")
    for cls in class_labels:
        print(f"{cls}: {per_class_auc[cls]:.3f}")
    print(f"Macro-AUC (final model, OvR): {macro_auc_final:.3f}")

    # ==========================
    # 7) Confusion matrix & per-class metrics
    # ==========================
    cm = confusion_matrix(y_true, y_pred, labels=class_labels)
    cm_df = pd.DataFrame(
        cm,
        index=[f"true_{c}" for c in class_labels],
        columns=[f"pred_{c}" for c in class_labels]
    )
    print("\nConfusion matrix:")
    print(cm_df)

    metrics = []
    total = cm.sum()

    for i, cls in enumerate(class_labels):
        TP = cm[i, i]
        FN = cm[i, :].sum() - TP
        FP = cm[:, i].sum() - TP
        TN = total - (TP + FN + FP)

        sensitivity = TP / (TP + FN) if (TP + FN) > 0 else np.nan  # recall
        specificity = TN / (TN + FP) if (TN + FP) > 0 else np.nan
        ppv = TP / (TP + FP) if (TP + FP) > 0 else np.nan  # precision
        f1 = 2 * ppv * sensitivity / (ppv + sensitivity) if (ppv + sensitivity) > 0 else np.nan

        metrics.append({
            "class": cls,
            "TP": TP,
            "FP": FP,
            "FN": FN,
            "TN": TN,
            "Sensitivity": sensitivity,
            "Specificity": specificity,
            "PPV": ppv,
            "F1": f1
        })

    metrics_df = pd.DataFrame(metrics)
    print("\nClass-specific metrics (test set):")
    print(metrics_df)

    # ---- Macro averages ----
    macro_sensitivity = metrics_df["Sensitivity"].mean()
    macro_specificity = metrics_df["Specificity"].mean()
    macro_ppv = metrics_df["PPV"].mean()
    macro_f1 = metrics_df["F1"].mean()

    print("\nMacro-averaged metrics (mean across classes):")
    print(f"Macro Sensitivity (Recall): {macro_sensitivity:.4f}")
    print(f"Macro Specificity:          {macro_specificity:.4f}")
    print(f"Macro PPV (Precision):      {macro_ppv:.4f}")
    print(f"Macro F1:                   {macro_f1:.4f}")

    print("\nSklearn classification report (per class):")
    print(classification_report(y_true, y_pred, labels=class_labels))

    # ==========================
    # 8) Metric explanation table (like XGBoost version)
    # ==========================

    # Map for explanation table
    your_metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Recall (Macro)": macro_sensitivity,
        "Precision (Macro)": macro_ppv,
        "Specificity (Macro)": macro_specificity,
        "F1-score (Macro)": macro_f1,
        "Macro-AUC": macro_auc_final,
        "Macro-F1": macro_f1
    }

    # Rename for consistency with earlier code
    metrics_expl = metrics_df.rename(columns={
        "Sensitivity": "Sensitivity",
        "Specificity": "Specificity",
        "PPV": "PPV",
        "F1": "F1"
    }).set_index("class")


    def format_per_class(col):
        """Return compact per-class string: inc/pers/rec/res (adapt to your label names)."""
        # Assuming your class labels are exactly these names:
        name_map = {
            "incidence": "inc",
            "persistence": "per",
            "recovery": "rec",
            "resilience": "res"
        }
        parts = []
        for cls in class_labels:
            short = name_map.get(cls, cls[:3])
            val = metrics_expl.loc[cls, col]
            parts.append(f"{short}:{val:.4f}")
        return ", ".join(parts)


    def format_per_class_auc():
        parts = []
        name_map = {
            "incidence": "inc",
            "persistence": "per",
            "recovery": "rec",
            "resilience": "res"
        }
        for cls in class_labels:
            short = name_map.get(cls, cls[:3])
            val = per_class_auc[cls]
            parts.append(f"{short}:{val:.4f}")
        return ", ".join(parts)


    rows = [
        ["Accuracy",
         "Percentage of all predictions that were correct.",
         "Overall correctness.",
         "Misleading with imbalance.",
         "≥ 0.70 (but limited meaning)",
         f"{your_metrics['Accuracy']:.4f}",
         "—"],

        ["Recall (Sensitivity)",
         "Of all actual cases, how many were detected?",
         "TP / (TP + FN)",
         "Critical for minority classes.",
         "≥ 0.40 acceptable",
         f"{your_metrics['Recall (Macro)']:.4f}",
         format_per_class("Sensitivity")],

        ["Precision (PPV)",
         "Of predicted positives, how many were correct?",
         "TP / (TP + FP)",
         "Avoids false alarms.",
         "≥ 0.40 acceptable",
         f"{your_metrics['Precision (Macro)']:.4f}",
         format_per_class("PPV")],

        ["Specificity",
         "Correctly identifies negatives.",
         "TN / (TN + FP)",
         "Prevents mislabeling into minority classes.",
         "≥ 0.80 good",
         f"{your_metrics['Specificity (Macro)']:.4f}",
         format_per_class("Specificity")],

        ["F1-score",
         "Harmonic mean of precision + recall.",
         "Balance of miss vs false alarm.",
         "Best single metric for imbalance.",
         "≥ 0.40 acceptable",
         f"{your_metrics['F1-score (Macro)']:.4f}",
         format_per_class("F1")],

        ["Macro-AUC",
         "Averaged AUC across classes.",
         "Ranking ability.",
         "Most robust for imbalance.",
         "≥ 0.70 publishable",
         f"{your_metrics['Macro-AUC']:.4f}",
         format_per_class_auc()],

        ["Macro-F1",
         "F1 averaged equally across classes.",
         "How well each class is predicted.",
         "Best fairness metric.",
         "≥ 0.45 acceptable",
         f"{your_metrics['Macro-F1']:.4f}",
         format_per_class("F1")]
    ]

    metrics_table_df = pd.DataFrame(rows, columns=[
        "Metric",
        "Meaning",
        "What It Measures",
        "Why It Matters",
        "Good Range",
        "Your Macro Result",
        "Per-Class Results"
    ])

    print("\n=== MODEL METRIC EXPLANATION TABLE (H2O AutoML LEADER) ===\n")
    print(metrics_table_df.to_string(index=False))



exit(1)
#####################################################################################################
#####################################################################################################
#####################################################################################################
#####################################################################################################
#####################################################################################################

print ("########################## Shap #################################################")

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

