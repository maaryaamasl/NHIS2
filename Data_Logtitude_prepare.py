import pandas as pd

# Load your merged dataset
df = pd.read_csv("Merged_data_19n20_ageEdu.csv")

# Define your variables
categorical_vars = ['MAXEDUC_A','EDUC_A','AGEP_A','REGION', 'ORIENT_A', 'MARITAL_A', 'RACEALLP_A', 'AGEP_A', 'EDUC_A', 'MAXEDUC_A', 'SEX_A', 'HISP_A', ]
numerical_vars = [ 'PHSTAT_A', 'ANXEV_A', 'DEPEV_A', 'BMICAT_A', 'ANXFREQ_A', 'ANXMED_A',
                  'DEPFREQ_A', 'DEPMED_A', 'PHQCAT_A', 'GADCAT_A', 'SMKCIGST_A', 'FAMINCTC_A',
                  'POVRATTC_A', 'INCGRP_A', 'RATCAT_A',   'NOTCOV_A',
                  'PAYBLL12M_A', 'PAYWORRY_A', 'MEDDL12M_A', 'RXSK12M_A', 'RXLS12M_A',
                  'RXDL12M_A', 'RXDG12M_A', 'MHTHDLY_A', 'MHTHND_A', 'EMPWRKLSWK_A',
                  'PCNTADTWKP_A', 'FDSCAT4_A', 'HOUYRSLIV_A', 'HOUTENURE_A',
                  'OPD12M_A', 'ARTHEV_A', 'CANEV_A', 'DIBEV_A', 'STREV_A', 'COPDEV_A',
                  'CHDEV_A', 'CHLEV_A', 'HYPEV_A', 'ANXLEVEL_A','URBRRL','FSNAP12M_A']

# -----------------------------------------------
# Create change features for numerical variables
# -----------------------------------------------
for var in numerical_vars:
    print("Num:",var)
    col_2019 = f"{var}_2019"
    col_2020 = f"{var}_2020"
    if col_2019 in df.columns and col_2020 in df.columns:
        # Calculate change
        df[f"change_{var}"] = df[col_2020] - df[col_2019]
        # Drop 2020 column
        df.drop(columns=[col_2020], inplace=True)
        # Rename 2019 column (remove suffix)
        df.rename(columns={col_2019: var}, inplace=True)

# -----------------------------------------------
# Create transition features for categorical variables
# -----------------------------------------------
for var in categorical_vars:
    print("Cat:",var)
    col_2019 = f"{var}_2019"
    col_2020 = f"{var}_2020"
    if col_2019 in df.columns and col_2020 in df.columns:
        # Detect changes only when different, else mark as 'NoChange'
        df[f"change_{var}"] = df.apply(
            lambda row: f"{row[col_2019]}_to_{row[col_2020]}" if row[col_2019] != row[col_2020] else "NoChange",
            axis=1
        )
        # Drop 2020 column
        df.drop(columns=[col_2020], inplace=True)
        # Rename 2019 column (remove suffix)
        df.rename(columns={col_2019: var}, inplace=True)

for column in df.columns:
            values = list(set(df[column]))
            if len(values) > 20:
                print(column, values[:20] + ['...'])
            else:
                print(column, values)

# Columns to drop (only 'NoChange'): ['change_URBRRL', 'change_REGION', 'change_ORIENT_A', 'change_RACEALLP_A', 'change_SEX_A', 'change_HISP_A']
cols_to_drop = [col for col in df.columns
                if df[col].nunique() == 1 and (df[col].unique()[0] == 'NoChange' or df[col].unique()[0] == 0)]
df.drop(columns=cols_to_drop, inplace=True)

print("### Columns to drop (only 'NoChange'):", cols_to_drop)

categorical_non_numeric = ['REGION', 'AGEP_A', 'ORIENT_A', 'MARITAL_A', 'RACEALLP_A', 'EDUC_A', 'MAXEDUC_A',
                           'change_MAXEDUC_A', 'change_EDUC_A', 'change_AGEP_A', 'change_MARITAL_A',
                           ]
print("### pd.get_dummies")
for column in categorical_non_numeric: #['REGION', 'ORIENT_A', 'MARITAL_A', 'RACEALLP_A', 'AGEP_A', 'EDUC_A', 'MAXEDUC_A']:
                                    # ['REGION', 'ORIENT_A', 'MARITAL_A', 'RACEALLP_A', 'AGEP_A', 'EDUC_A', 'MAXEDUC_A', 'SEX_A', 'HISP_A','MAXEDUC_A','EDUC_A','AGEP_A', ]
    df_dummy = pd.get_dummies(df[column], prefix=(column + "_"))
    df = pd.concat([df, df_dummy], axis=1)
    df.drop(column, axis=1, inplace=True)

# Drop those columns

print("### set(df[column])")
for column in df.columns:
            values = list(set(df[column]))
            if len(values) > 20:
                print(column, values[:20] + ['...'])
            else:
                print(column, values)

summary = {}
def get_count_and_percentage(column):
    count = column.value_counts()
    percentage = column.value_counts(normalize=True) * 100
    result = pd.DataFrame({'Count': count, 'Percentage': percentage})
    return result
for col in df.columns:
    result = get_count_and_percentage(df[col])
    print(f"=== {col} ===")
    print(result)


# 2019
# URBRRL_2019 HISP_A_2019 SEX_A_2019 FSNAP12M_A_2019
# 2020
# URBRRL_2020 HISP_A_2020 SEX_A_2020 FSNAP12M_A_2020
# 0/1
# change_ANXEV_A change_DEPEV_A change_ANXMED_A change_DEPMED_A change_NOTCOV_A change_PAYBLL12M_A change_MEDDL12M_A change_RXDG12M_A change_MHTHDLY_A change_MHTHND_A change_EMPWRKLSWK_A

df.to_csv("Data_longtitude_ageEdu.csv", index=False)


