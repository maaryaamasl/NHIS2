import pandas as pd

# Read the files
df_2019 = pd.read_csv("Cleaned_data_2019_ageEdu.csv")
print("df_2019.shape: ",df_2019.shape)
df_2020 = pd.read_csv("Cleaned_data_2020_ageEdu.csv")
print("df_2020.shape: ",df_2020.shape)

hhx_map = pd.read_csv("./adultlong20.csv")
hhx_map['HHX_2019'] = hhx_map['HHX_2019'].astype(str).str.strip()
hhx_map['HHX_2020'] = hhx_map['HHX_2020'].astype(str).str.strip()

df_2019['HHX'] = df_2019['HHX'].astype(str).str.strip()#.str.lstrip('0')
df_2020['HHX'] = df_2020['HHX'].astype(str).str.strip()#.str.lstrip('0')

print("Unique HHX in 2019:", df_2019['HHX'].nunique())
print("Unique HHX in 2020:", df_2020['HHX'].nunique())

shared_HHX = set(df_2019['HHX']) & set(df_2020['HHX'])
print("before Shared HHX count:", len(shared_HHX))

hhx_dict = dict(zip(hhx_map['HHX_2019'], hhx_map['HHX_2020']))
hhx_dict_2020 = dict(zip( hhx_map['HHX_2020'], hhx_map['HHX_2019']))

df_2019['HHX_original'] = df_2019['HHX']
df_2019['HHX_mapped'] = df_2019['HHX'].map(hhx_dict)
unmapped = df_2019[df_2019['HHX_mapped'].isna()]
print(f"Unmapped HHX count: {len(unmapped)}")
print(unmapped['HHX_original'].unique())
mapped = df_2019[~df_2019['HHX_mapped'].isna()]
print(f"Mapped HHX count: {len(mapped)}")
print(mapped['HHX_original'].unique())

df_2020['HHX_original'] = df_2020['HHX']
df_2020['HHX_mapped'] = df_2020['HHX'].map(hhx_dict_2020)
unmapped_2020 = df_2020[df_2020['HHX_mapped'].isna()]
print(f"Unmapped HHX count 2020: {len(unmapped_2020)}")
print(unmapped_2020['HHX_original'].unique())
mapped_2020 = df_2020[~df_2020['HHX_mapped'].isna()]
print(f"Mapped HHX count 2020: {len(mapped_2020)}")
print(mapped_2020['HHX_original'].unique())

df_2019['HHX'] = df_2019['HHX'].map(hhx_dict)
unmapped_count = df_2019['HHX'].isna().sum()
print(f"Number of HHX values in 2019 that could not be mapped to 2020: {unmapped_count}")



shared_HHX = set(df_2019['HHX']) & set(df_2020['HHX'])
print("after Shared HHX count:", len(shared_HHX))


df_2019 = df_2019.dropna(subset=['HHX'])

# Perform inner join on 'HHX'
merged_df = pd.merge(df_2019, df_2020, on='HHX', how='inner', suffixes=('_2019', '_2020'))
print("Merged shape:", merged_df.shape)
print(merged_df.head())


# exit()



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

for column in merged_df.columns:
            values = list(set(merged_df[column]))
            if len(values) > 20:
                print(column, values[:20] + ['...'])
            else:
                print(column, values)

merged_df.drop(['HHX_original_2020', 'HHX_mapped_2020','High_impact_chronic_pain_2020','HHX_original_2019', 'HHX_mapped_2019','High_impact_chronic_pain_2019','HHX'], axis=1, inplace=True) # not needed #############
merged_df.to_csv('Merged_data_19n20_ageEdu.csv', index=False, header=True)