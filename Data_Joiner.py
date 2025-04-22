import pandas as pd

# Read the files
df_2019 = pd.read_csv("Cleaned_data_2019_ageEdu.csv")
print("df_2019.shape: ",df_2019.shape)
df_2020 = pd.read_csv("Cleaned_data_2020_ageEdu.csv")
print("df_2020.shape: ",df_2020.shape)


##############
print("Unique HHX in 2019:", df_2019['HHX'].nunique())
print("Unique HHX in 2020:", df_2020['HHX'].nunique())

# Check overlap between the two
shared_HHX = set(df_2019['HHX']) & set(df_2020['HHX'])
print("Shared HHX count:", len(shared_HHX))

# Check for data types and formatting
print("2019 HHX dtype:", df_2019['HHX'].dtype)
print("2020 HHX dtype:", df_2020['HHX'].dtype)

# Example of fixing formatting:
df_2019['HHX'] = df_2019['HHX'].astype(str).str.strip()#.str.lstrip('0')
df_2020['HHX'] = df_2020['HHX'].astype(str).str.strip()#.str.lstrip('0')


# Perform inner join on 'HHX'
merged_df = pd.merge(df_2019, df_2020, on='HHX', how='inner', suffixes=('_2019', '_2020'))
print("Merged shape:", merged_df.shape)
print(merged_df.head())


# exit()

for column in merged_df.columns:
            values = list(set(merged_df[column]))
            if len(values) > 20:
                print(column, values[:20] + ['...'])
            else:
                print(column, values)

def classify_pain(row):
    pain_2019 = row['High_impact_chronic_pain_2019']
    pain_2020 = row['High_impact_chronic_pain_2020']
    
    if pain_2019 == 1 and pain_2020 == 1:
        return 'persistence'
    elif pain_2019 == 0 and pain_2020 == 1:
        return 'incidence'
    elif pain_2019 == 1 and pain_2020 == 0:
        return 'recovery'
    elif pain_2019 == 0 and pain_2020 == 0:
        return 'resilience'
    else:
        return 'unknown'

merged_df['pain_trajectory'] = merged_df.apply(classify_pain, axis=1)
counts = merged_df['pain_trajectory'].value_counts()
# Percentages (rounded to 2 decimals)
percentages = merged_df['pain_trajectory'].value_counts(normalize=True).round(4) * 100
# Combine into a single DataFrame
pain_stats = pd.DataFrame({
    'Count': counts,
    'Percentage (%)': percentages
})
print(pain_stats)

def get_count_and_percentage(column):
                count = column.value_counts()
                percentage = column.value_counts(normalize=True) * 100
                result = pd.DataFrame({'Count': count, 'Percentage': percentage})
                return result

for col in ['High_impact_chronic_pain_2019','High_impact_chronic_pain_2020']: # 'Chronic_Pain_2019', 'Chronic_Pain_2020',
            result = get_count_and_percentage(merged_df[col])
            print(f"=== {col} ===")
            print(result)
            print("\n")