import pandas as pd

# Read the files
df_2019 = pd.read_csv("Cleaned_data_2019_ageEdu.csv")
df_2020 = pd.read_csv("Cleaned_data_2020_ageEdu.csv")

# Perform inner join on 'HHX'
merged_df = pd.merge(df_2019, df_2020, on='HHX', how='inner', suffixes=('_2019', '_2020'))

# Optional: check the result
print("Merged shape:", merged_df.shape)
print(merged_df.head())

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

for col in ['Chronic_Pain_2019','High_impact_chronic_pain_2019','Chronic_Pain_2020','High_impact_chronic_pain_2020']:
            result = get_count_and_percentage(merged_df[col])
            print(f"=== {col} ===")
            print(result)
            print("\n")