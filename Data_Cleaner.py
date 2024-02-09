# This is a sample Python script.
import numpy as np
import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
import sklearn as sk

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press ⌘F8 to toggle the breakpoint.

    ################################################################################
    df = pd.read_csv("./adult19.csv")

    variable_list_df = pd.read_excel('NHIS variable list_Modified.xlsx')
    selected_columns = variable_list_df['variable(s)'].tolist()
    print("selected_columns",selected_columns, '\nlen', len(selected_columns))
    selected_columns = [x.upper() for x in selected_columns]

    print("\nMissing Columns: ", end=" ")
    for x in selected_columns:
        try:
            _ = df[x]
        except:
            print(x, end=" ,")
    print()
    # 2021
    # Missing Columns:  EMPDYSMSS2_A ,PHQCAT_A ,GADCAT_A ,FAMINCTC_A ,INCGRP_A ,EDUC_A ,MAXEDUC_A ,MEDDNG12M_A ,
    # EMPWRKLSWK_A ,EMPWKHRS2_A ,EMPWRKFT_A ,FSNAP12M_Z ,OPD12M_A ,OPD3M_A ,OPDACUTE_A ,OPDCHRONIC_A ,OPDFREQ_A ,
    # 2019
    # Missing Columns:  MEDDNG12M_A ,FSNAP12M_Z , <<<<<<<<<<<<<<<<<<<<<
    missing_column = ['MEDDNG12M_A', 'FSNAP12M_Z']
    selected_columns = [x for x in selected_columns if x not in missing_column]
    selected_data = df[selected_columns].copy()

    # print("\nSelected Data:")
    # print(selected_data.head())
    # # print("Info.\n", selected_data.info())
    # print("Desc. Data.\n", selected_data.describe(include='all'))
    # print("Desc. OutCome\n", selected_data[['PAIFRQ3M_A','PAIAMNT_A','PAIWKLM3M_A','PAIAFFM3M_A']].describe(include='all'))

    ### NOTE: Finalize outcome ###
    print("\nFinalize outcome")
    print("Number of rows raw:", len(selected_data))
    missingCount = pd.DataFrame([selected_data.isna().sum().values], columns=selected_data.columns.values)
    print(missingCount[['PAIFRQ3M_A','PAIAMNT_A','PAIWKLM3M_A','PAIAFFM3M_A']])
    outcomes = selected_data[['PAIFRQ3M_A','PAIWKLM3M_A']].fillna('Unknown')
    selected_data[['PAIWKLM3M_A']] = selected_data[['PAIWKLM3M_A']].fillna(0)
    for column in ['PAIFRQ3M_A','PAIWKLM3M_A']:
        print(outcomes[column].value_counts()) # column,set(outcomes[column]),
    selected_data.dropna(subset=['PAIFRQ3M_A','PAIWKLM3M_A'], inplace=True) # 'PAIAMNT_A', ,'PAIAFFM3M_A'
    missingCount = pd.DataFrame([selected_data.isna().sum().values], columns=selected_data.columns.values)
    print(missingCount[['PAIFRQ3M_A','PAIWKLM3M_A']])
    print("Number of rows after first drop NaN outcome:", len(selected_data))
    # Chronic pain = Pain reported on most days or every day during the previous 3 months.
    # PAIFRQ3M_A as our primary outcome:
    # 1 Never, 2 Some days, 3 Most days, 4 Every day, 7 Refused, 8 Not Ascertained, 9 Don't Know
    # Chronic Pain = 3 or 4 # No Chronic pain = 1 or 2 # And missing = 7, 8, or 9
    print("outcome convert to int")
    selected_data.drop(['PAIAMNT_A', 'PAIAFFM3M_A'], axis=1, inplace=True)  # not needed #############
    for column in ['PAIFRQ3M_A', 'PAIWKLM3M_A']:
        selected_data[column] = selected_data[column].astype(int)
    mapping_dict = {
        'PAIFRQ3M_A': {1 : 0, 2 : 0, 3 : 1, 4 : 1, 7 : np.nan, 8 : np.nan, 9 : np.nan}, # zero, no, 1, yes
        'PAIWKLM3M_A': {1 : 0, 2 : 0, 3 : 1, 4 : 1, 7 : np.nan, 8 : np.nan, 9 : np.nan}
    }
    print("outcome map & dropna 2")
    selected_data.replace(mapping_dict, inplace=True)
    selected_data.dropna(subset=['PAIFRQ3M_A', 'PAIWKLM3M_A'], inplace=True,how='any')
    print("Outcomes After map and dropna 2")
    outcomes = selected_data[['PAIFRQ3M_A', 'PAIWKLM3M_A']]
    for column in ['PAIFRQ3M_A', 'PAIWKLM3M_A']:
        print(column, set(outcomes[column]), selected_data[column].value_counts().values)
    print("Number of rows after 2nd map&dropna:", len(selected_data))
    # exit(-1)
    selected_data['Chronic_Pain'] = selected_data['PAIFRQ3M_A']
    # High-impact chronic pain = Chronic pain that limited life or work activities on most days or every day during the previous 3 months.
    # (combination of PAIFRQ3M_A and paiwklm3m_a) as secondary outcome
    # PAIFRQ3M_A = 3 or 4  AND. paiwklm3m_a = 3 or 4
    selected_data['High_impact_chronic_pain'] = (selected_data['PAIFRQ3M_A'] == 1) & (selected_data['PAIWKLM3M_A'] == 1)
    selected_data.drop(['PAIFRQ3M_A', 'PAIWKLM3M_A'], axis=1, inplace=True) # not needed #############
    mapping_dict = {'High_impact_chronic_pain': {True: 1, False: 0}}
    print("outcome map & dropna 2")
    selected_data.replace(mapping_dict, inplace=True)
    for column in ['Chronic_Pain', 'High_impact_chronic_pain']:
        print(column, set(selected_data[column]), selected_data[column].value_counts().values)
    # exit(-1)

    ### NOTE: Missing ###
    missingCount = pd.DataFrame([selected_data.isna().sum().values], columns=selected_data.columns.values)
    print("\nMissing Count: \n", missingCount)
    len_data = selected_data.shape[0]
    print("selected_data", selected_data.shape)
    missing30p = missingCount.loc[:, missingCount.apply(lambda s: s >= (len_data*0.30)).to_numpy().flatten()]
    print("\n Missing 30% Plus:",missing30p.shape[1],"\n",missing30p)
    selected_columns = [x for x in selected_data.columns if x not in missing30p.columns]
    selected_data = selected_data[selected_columns]
    missing30m = missingCount.loc[:, missingCount.apply(lambda s: s < len_data * 0.30).to_numpy().flatten()]
    missing30m = missing30m.loc[:, missing30m.apply(lambda s: s > 0).to_numpy().flatten()]
    print("\n Missing 30 Minus:", missing30m.shape[1],"\n", missing30m )
    print("selected_data", selected_data.shape)
    # exit(-1)

    print("\nValue Sets")
    interim_data = selected_data.fillna('Unknown') # Affects: OPD12M_A (9g)  RXDL12M_A (9g) RXLS12M_A (9g) RXSK12M_A (9g) MAXEDUC_A (X) ANXLEVEL_A (9g)
    for column in selected_data.columns:
        print(column, set(interim_data[column]))
    # exit(-1)

    # mapping Dict.
    mapping_dict_var = {'URBRRL': {1: 4, 2: 3, 3: 2, 4: 1},
                        # 'URBRRL': {1: 'Large central metro', 2: 'Large fringe metro', 3: 'Medium and small metro', 4: 'Nonmetropolitan'},
                        'REGION': {1: 'Northeast', 2: 'Midwest', 3: 'South', 4: 'West'},
                        'ASTATNEW': {1: 1, 5: 0},  # {1: 'Completed', 5: 'Sufficient Partial'}
                        'AGEP_A': {97: np.nan, 99: np.nan},  # Refused, Don't Know # age 18 to 85+
                        'HISP_A': {1: 1, 2: 0},
                        'PHSTAT_A': {1: 4, 2: 3, 3: 2, 4: 1, 5: 0, 7: np.nan, 9: np.nan},
                        # 1: 'Excellent', 2: 'Very Good', 3: 'Good', 4: 'Fair', 5: 'Poor', Refused, Don't Know
                        'ORIENT_A': {1: 'GayLesbian', 2: 'Straight', 3: 'Bisexual', 4: 'Unknown', 5: 'Unknown',
                                     7: 'Unknown', 8: 'Unknown'},
                        # 4:something else 5: 'I don\'t know the answer', 7: 'Unknown', 8: 'Unknown'
                        'MARITAL_A': {1: 'Married', 2: 'Unmarried couple', 3: 'Neither', 7: 'Unknown', 8: 'Unknown'},
                        'RACEALLP_A': {1: 'White', 2: 'Black/African-American', 3: 'Asian', 4: 'AIAN',
                                       5: 'AIAN and any other group', 6: 'Other single and multiple races',
                                       8: 'Unknown'},
                        'HISPALLP_A': {1: 'Hispanic', 2: 'NH White', 3: 'NH Black/African-American', 4: 'NH Asian',
                                       5: 'NH AIAN', 6: 'NH AIAN and any other group',
                                       7: 'Other single and multiple races'},
                        'SEX_A': {1: 1, 2: 0, 7: np.nan},  # M/F
                        'ANXEV_A': {1: 1, 2: 0, 7: np.nan, 9: np.nan},  # Y/N
                        'DEPEV_A': {1: 1, 2: 0, 7: np.nan, 9: np.nan},  # Y/N
                        'BMICAT_A': {9: np.nan},  # 1: 'Underweight', 2: 'Healthy weight',..., Overweight,Obese
                        'ANXFREQ_A': {1: 4, 2: 3, 3: 2, 4: 1, 5: 0, 7: np.nan, 9: np.nan},
                        # 1:Daily, 2:Weekly, ..., 5:Never
                        'ANXMED_A': {1: 1, 2: 0, 7: np.nan, 9: np.nan},  # Y/N
                        'ANXLEVEL_A': {1: 1, 2: 3, 3: 2, 7: np.nan, 9: np.nan, 'Unknown': np.nan},
                        # 1: little, 2 alot, 3 somewhere between
                        'DEPFREQ_A': {1: 4, 2: 3, 3: 2, 4: 1, 5: 0, 7: np.nan, 9: np.nan},
                        # 1:Daily, Weekly, Monthly, afew year, never
                        'DEPMED_A': {1: 1, 2: 0, 7: np.nan, 9: np.nan},  # Y/N
                        'PHQCAT_A': {8: np.nan},  # 1: none, 2: mild, 3: moderate, etc.
                        'GADCAT_A': {8: np.nan},  # 1: none, 2: mild, 3: moderate, etc.
                        'SMKCIGST_A': {1: 3, 2: 2, 3: 1, 4: 0, 5: np.nan, 9: np.nan},
                        # 1:everyday, 2: someday, 4: never
                        # 'FAMINCTC_A': {1: '', 2: '', 3: '', 4: '', 5: '', 7: np.nan, 8: np.nan, 9: np.nan}, # Numeric
                        # 'POVRATTC_A': {1: '', 2: '', 3: '', 4: '', 5: '', 7: np.nan, 8: np.nan, 9: np.nan},  # Numeric
                        # 'INCGRP_A': {1: '', 2: '', 3: '', 4: '', 5: '', 7: np.nan, 8: np.nan, 9: np.nan},  # Numeric
                        # 'RATCAT_A': {1: '', 2: '', 3: '', 4: '', 5: '', 7: np.nan, 8: np.nan, 9: np.nan},  # Numeric
                        'EDUC_A': {97: np.nan, 99: np.nan},  # education from never 0 to 11 doctoral
                        'MAXEDUC_A': {'Unknown': np.nan},
                        'NOTCOV_A': {1: 1, 2: 0, 9: np.nan},  # 1: not cov, 2: covered (health)
                        'MEDICARE_A': {1: 1, 2: 1, 3: 0, 7: np.nan, 8: np.nan, 9: np.nan},
                        # 1:Yes, 2:yes no info, 3: no,
                        'MEDICAID_A': {1: 1, 2: 1, 3: 0, 7: np.nan, 8: np.nan, 9: np.nan},
                        # 1:Yes, 2:yes no info, 3: no,
                        'PRIVATE_A': {1: 1, 2: 1, 3: 0, 7: np.nan, 8: np.nan, 9: np.nan},
                        # 1:Yes, 2:yes no info, 3: no,
                        'CHIP_A': {1: 1, 2: 1, 3: 0, 7: np.nan, 8: np.nan, 9: np.nan},  # 1:Yes, 2:yes no info, 3: no,
                        'OTHPUB_A': {1: 1, 2: 1, 3: 0, 7: np.nan, 8: np.nan, 9: np.nan},  # 1:Yes, 2:yes no info, 3: no,
                        'OTHGOV_A': {1: 1, 2: 1, 3: 0, 7: np.nan, 8: np.nan, 9: np.nan},  # 1:Yes, 2:yes no info, 3: no,
                        'MILITARY_A': {1: 1, 2: 1, 3: 0, 7: np.nan, 8: np.nan, 9: np.nan},
                        # 1:Yes, 2:yes no info, 3: no,
                        'HICOV_A': {1: 1, 2: 0, 7: np.nan, 8: np.nan, 9: np.nan},  # 1:Yes, 2:No,
                        'PAYBLL12M_A': {1: 1, 2: 0, 7: np.nan, 8: np.nan, 9: np.nan},  # 1:Yes, 2:No,
                        'PAYWORRY_A': {1: 2, 2: 1, 3: 0, 7: np.nan, 8: np.nan, 9: np.nan},
                        # 1:very worries, 2:somewhat, 3:not at all
                        'MEDDL12M_A': {1: 1, 2: 0, 7: np.nan, 8: np.nan, 9: np.nan},  # 1:Yes, 2:No,
                        'RXSK12M_A': {1: 1, 2: 0, 7: np.nan, 8: np.nan, 9: np.nan},  # 1:Yes, 2:No,
                        'RXLS12M_A': {1: 1, 2: 0, 7: np.nan, 8: np.nan, 9: np.nan},  # 1:Yes, 2:No,
                        'RXDL12M_A': {1: 1, 2: 0, 7: np.nan, 8: np.nan, 9: np.nan},  # 1:Yes, 2:No,
                        'RXDG12M_A': {1: 1, 2: 0, 7: np.nan, 8: np.nan, 9: np.nan},  # 1:Yes, 2:No,
                        'MHTHDLY_A': {1: 1, 2: 0, 7: np.nan, 8: np.nan, 9: np.nan},  # 1:Yes, 2:No,
                        'MHTHND_A': {1: 1, 2: 0, 7: np.nan, 8: np.nan, 9: np.nan},  # 1:Yes, 2:No,
                        'EMPWRKLSWK_A': {1: 1, 2: 0, 7: np.nan, 8: np.nan, 9: np.nan},  # 1:Yes, 2:No,
                        'PCNTADTWKP_A': {8: np.nan},  # number of adults working
                        'FDSCAT4_A': {8: np.nan},  # the higher the value higher food security
                        'HOUYRSLIV_A': {7: np.nan, 8: np.nan, 9: np.nan},
                        # higher value, higher years in apartment/hourse
                        'HOUTENURE_A': {1: 1, 2: 0, 3: np.nan, 7: np.nan, 8: np.nan, 9: np.nan}, # 1: 'Owned', 2: 'Rented'
                        'PAIBACK3M_A': {1: 0, 2: 1, 3: 3, 4: 2, 7: np.nan, 8: np.nan, 9: np.nan},
                        # 1: not at all, 2:a little, 3:a lot, 4:between a little and alot
                        'PAIULMB3M_A': {1: 0, 2: 1, 3: 3, 4: 2, 7: np.nan, 8: np.nan, 9: np.nan},
                        # 1: not at all, 2:a little, 3:a lot, 4:between a little and alot
                        'PAILLMB3M_A': {1: 0, 2: 1, 3: 3, 4: 2, 7: np.nan, 8: np.nan, 9: np.nan},
                        # 1: not at all, 2:a little, 3:a lot, 4:between a little and alot
                        'PAIHDFC3M_A': {1: 0, 2: 1, 3: 3, 4: 2, 7: np.nan, 8: np.nan, 9: np.nan},
                        # 1: not at all, 2:a little, 3:a lot, 4:between a little and alot
                        'PAIAPG3M_A': {1: 0, 2: 1, 3: 3, 4: 2, 7: np.nan, 8: np.nan, 9: np.nan},
                        # 1: not at all, 2:a little, 3:a lot, 4:between a little and alot
                        'PAITOOTH3M_A': {1: 0, 2: 1, 3: 3, 4: 2, 7: np.nan, 8: np.nan, 9: np.nan},
                        # 1: not at all, 2:a little, 3:a lot, 4:between a little and alot
                        'OPD12M_A': {1: 1, 2: 0, 7: np.nan, 8: np.nan, 9: np.nan, 'Unknown': np.nan},  # 1:Yes, 2:No,
                        }
    for col in selected_data.columns:
        selected_data[col] = pd.to_numeric(selected_data[col], errors='coerce', downcast='integer') # selected_data[col].astype(int) #
    selected_data.replace(mapping_dict_var, inplace=True)
    print("\nValue after Dict. & Map Features")
    for column in selected_data.columns:
        print(column, set(selected_data[column]))
        # selected_data[column] = pd.to_numeric(selected_data[column], errors='coerce') # selected_data[[column]].astype(int)
        # selected_data[column] = selected_data[column].replace(mapping_dict) # , inplace=True
        # print(column, set(selected_data[column]))
    # exit(-1)

    print("######### Before categorization###########")
    def get_count_and_percentage(column):
        count = column.value_counts()
        percentage = column.value_counts(normalize=True) * 100
        result = pd.DataFrame({'Count': count, 'Percentage': percentage})
        return result

    for col in selected_data.columns:
        result = get_count_and_percentage(selected_data[col])
        print(f"=== {col} ===")
        print(result)
        print("\n")
    # Imputation # replace with median
    for column in ['AGEP_A', 'PHSTAT_A', 'ANXEV_A', 'DEPEV_A', 'BMICAT_A', 'ANXFREQ_A', 'ANXMED_A',  'DEPFREQ_A', 'DEPMED_A', 'PHQCAT_A',
                   'GADCAT_A', 'SMKCIGST_A', 'FAMINCTC_A', 'POVRATTC_A', 'INCGRP_A', 'RATCAT_A', 'EDUC_A', 'MAXEDUC_A', 'NOTCOV_A', 'MEDICARE_A', 'MEDICAID_A',
                   'PRIVATE_A', 'CHIP_A', 'OTHPUB_A', 'OTHGOV_A', 'MILITARY_A', 'HICOV_A', 'PAYBLL12M_A', 'PAYWORRY_A', 'MEDDL12M_A','RXSK12M_A','RXLS12M_A',
                   'RXDL12M_A', 'RXDG12M_A', 'MHTHDLY_A', 'MHTHND_A', 'EMPWRKLSWK_A', 'PCNTADTWKP_A' ,'FDSCAT4_A' ,'HOUYRSLIV_A', 'HOUTENURE_A',
                    'OPD12M_A']: # 'ANXLEVEL_A', 'PAIBACK3M_A','PAIULMB3M_A', 'PAILLMB3M_A', 'PAIHDFC3M_A', 'PAIAPG3M_A', 'PAITOOTH3M_A'
        median_value = selected_data[column].median()
        selected_data[column].fillna(median_value, inplace=True)

    # Drop Rows with missing
    print("no drop sex", len(selected_data))
    selected_data.dropna(subset=['SEX_A'], inplace=True)
    print("drop sex", len(selected_data))


    # Make categorical
    for column in ['REGION', 'ORIENT_A', 'MARITAL_A', 'RACEALLP_A', 'HISPALLP_A',]:
        df_dummy = pd.get_dummies(selected_data[column], prefix=(column+"_"))
        selected_data = pd.concat([selected_data, df_dummy], axis=1)
        selected_data.drop(column, axis=1, inplace=True)

    print("\nAll done")
    for column in selected_data.columns:
        print(column, set(selected_data[column]))

    print("\nWrite to file")
    selected_data.to_csv('Cleaned_data_2019.csv', index=False, header=True)
    print(selected_data.describe())

    print("######### After categorization###########")
    def get_count_and_percentage(column):
        count = column.value_counts()
        percentage = column.value_counts(normalize=True) * 100
        result = pd.DataFrame({'Count': count, 'Percentage': percentage})
        return result

    for col in selected_data.columns:
        result = get_count_and_percentage(selected_data[col])
        print(f"=== {col} ===")
        print(result)
        print("\n")
    ################################################################################



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
