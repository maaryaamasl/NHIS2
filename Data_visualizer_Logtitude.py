
import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import re

# TODO: * "" * "HISPALLP_A__NH Black-African-American-1" * "HISPALLP_A__NH White-1" * "SEX_A-1" - "SEX_A-0"

take_abs = True
shap_reason = "pain_trajectory"
shap_dir = f"C:/venv-Shap/{shap_reason}/"


# variable_list_df = pd.read_excel('./NHIS variable list_Modified_new.xlsx')
variable_list_df = pd.read_excel('NHIS variable list REVISED 6-14-24_Now_Moddified new.xlsx')
variable_list_df['category'] = variable_list_df['category'].apply(lambda x: x.title() if isinstance(x, str) else x)
selected_columns = variable_list_df['variable(s)'].tolist()

column_desc = dict(zip(variable_list_df['variable(s)'].str.upper(),variable_list_df['description']))
column_cat = dict(zip(variable_list_df['variable(s)'].str.upper(), variable_list_df['category']))
print("column_desc ",len(column_desc), column_desc)
print("column_cat ",len(column_cat), column_cat)
print("cats:", set(variable_list_df['category']))

feature_names = pd.read_csv(shap_dir+ 'columns.csv')['Column Names'].values
class_names = pd.read_csv(shap_dir+  'class_names.csv')['Class Names'].values
num_samples = int(np.loadtxt(shap_dir+ 'shape.csv'))
num_features = len(feature_names)
num_classes = len(class_names)
print(f"Samples: {num_samples}, Features: {num_features}, Classes: {num_classes}")

shap_values_array = np.zeros((num_samples, num_features, num_classes))
print("load")
for i in range(num_samples):
    print('\\shap_' + str(i) + ".csv")
    shap_values_array[i] = np.loadtxt(shap_dir+f'shap_{i}.csv')
print("loaded")
print('\nD1 samples:', len(shap_values_array), '\nD2 features:', len(shap_values_array[0]),'\nD2 classes:', len(shap_values_array[0][0]) )#,'\nD3 Columns/features:',len(shap_values[0][0]),'\nvalue:',shap_values[0][0][0])
average_shap_values =0
if take_abs:
    average_shap_values = np.mean(np.abs(shap_values_array), axis=0)
else:
    average_shap_values = np.mean(shap_values_array, axis=0)
print("average_shap_values shape", average_shap_values.shape)


df = pd.DataFrame(average_shap_values, columns=class_names,index=feature_names)
print(df.head(5),"\n",str(list(df.index)),"\n", len(list(df.index)))



for class_idx, class_name in enumerate(class_names):
    print(f"\nPlotting for class: {class_name}")
    sorted_df = df.sort_values(by=class_name, ascending=False).copy()

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

    # Function to create labels
    def create_label(inx):
        Change = False
        if inx in custom_labels:
            return custom_labels[inx]
        else:
            if inx.startswith('change_'):
                change = True
                # print("change = True")
                inx = inx.replace('change_', '')
                print(inx)
                parts = inx.split('__')
                # parts_change = inx.split('change_') # len(parts_change) = 2
                if len(parts) == 2:
                    print(f"Change in: {column_desc.get(parts[0], '')} [{parts[1].replace('_', ' ')}]")
                    return f"Change in: {column_desc.get(parts[0], '')} [{parts[1].replace('_', ' ')}]"  # .capitalize()
                else:
                    print(f"Change in: {column_desc.get(parts[0], '')} [{(str(parts[0]))}]")
                    return f"Change in: {column_desc.get(parts[0], '')} [{(str(parts[0]))}]"
            else:
                parts = inx.split('__')
                # parts_change = inx.split('change_') # len(parts_change) = 2
                if len(parts) == 2:
                    return f"{column_desc.get(parts[0], '')} [{(str(parts[0]))}] ({(str(parts[1]))})" # .capitalize()
                else:
                    return f"{column_desc.get(parts[0], '')} [{(str(parts[0]))}]"
                # return inx.str.split('__', expand=True).apply(lambda x: f"{column_desc.get(x[0], '')} [{(str(x[0]).capitalize())}] ({(str(x[1]).capitalize())})", axis=1)


    df[['label','cat','color']] = np.nan
    df["inx"]=df.index
    print("df['inx'].head():::\n",df["inx"].tail(5))
    df['label'] = df["inx"].apply(create_label)#[0].map(column_desc)  ######## .str.split('__', expand=True).apply(lambda x: f"{column_desc.get(x[0], '')} [{(str(x[0]).capitalize())}] ({(str(x[1]).capitalize())})", axis=1)
    print("df['label'].head():::\n",df['label'].tail(5))
    # df['label'] = df['label'].str.replace('(None)', '')

    df['cat'] = df["inx"].apply(lambda x: x.replace('change_', '') if x.startswith('change_') else x)
    df['cat'] = df['cat'].str.split('__', expand=True)[0].map(column_cat)

    df = df[~df['cat'].isin(['nan', 'Filter', np.nan])]  # drop filter
    df = df[~df['cat'].isin(['nan', 'Filter', np.nan])]
    df = df[~df["inx"].isin(['MARITAL_A__9'])]
    print("set(df['cat'].unique())", set(df['cat'].unique()))
    df['color'] = df.cat.map({'risk factor':1, 'covariate':2, 'filter':3, 'risk factor and moderator':4, 'SES':5 })
    df['color_label'] = df['cat']
    df = df.sort_values(by=class_name, ascending=False, )
    # df = df[~df['category'].isin(['nan', 'filter'])] # drop filter
    # print(df.head(75),"\n")
    # df['label'] = df['label'].str.replace("  "," ").replace("\n"," ").replace("\t"," ")

    palette = sns.color_palette("bright", 10) # pastel
    palette = {"Geographic": palette[1], "Socioeconomic Position": palette[3], #"Primary Outcome": palette[0],
               "Demographic": palette[9] , 'Physical Health': palette[4], 'Mental Health': palette[2]}  # 7 grey 5 dark red
    hue_order = ['Geographic', 'Socioeconomic Position',  'Demographic', 'Physical Health', 'Mental Health'] # 'Primary Outcome',

    df_filtered = df.copy()
    df_filtered['index_df'] = df.index
    # df_filtered[class_name] = df_filtered[class_name]#.abs()
    df_filtered = df_filtered.sort_values(by=class_name, ascending=False).reset_index(drop=True)
    df_filtered['label']= df_filtered['label'].apply(lambda x: x if isinstance(x, str) else x) # .capitalize()
    df_filtered['label']= (df_filtered['label']
                           .apply(lambda x: x.replace(" (none)","")) #.replace(r"\(none\)", "", regex=True)
                           .replace(r"nh ", "non-hispanic ", regex=True) #.replace("nh ", "Non-Hispanic")
                           # .apply(lambda x: re.sub(r'\[.*?\]', '', x))
                           .apply(lambda x: x.replace("  "," "))
                           .apply(lambda x: x.replace("(gad)", ""))  #
                           .apply(lambda x: x.replace("(phq)", ""))#
                           .apply(lambda x: x.replace("Medicaid recode", "Medicaid"))  #
                           .apply(lambda x: x.replace(" recode", ""))
                           .apply(lambda x: x.replace("(neither)", "(not married or living with a partner as an unmarried)"))
                           .apply(lambda x: x.replace("Get sik", "Get sick")) #
                           .apply(lambda x: x.replace("or living with a partner as an unmarried", "or living with a partner and unmarried"))
                           .apply(lambda x: x.replace("Us ", "US "))
                           .apply(lambda x: x.replace("gaylesbian", "gay or lesbian"))
                           .apply(lambda x: x.replace("chip", "CHIP"))
                           .apply(lambda x: x.replace("Other government program", "Other government insurance program"))
                           .apply(lambda x: x.replace("non-hispanic white", "Non-Hispanic White"))
                           .apply(lambda x: x.replace("non-hispanic black/african-american", "Non-Hispanic Black/African-American"))
                           .apply(lambda x: x.replace("black/african-american", "Black/African-American"))
                           .apply(lambda x: x.replace("white", "White"))
                           .apply(lambda x: x.replace("asian", "Asian"))
                           .apply(lambda x: x.replace("non-hispanic asian", "Non-Hispanic Asian"))
                           .apply(lambda x: x.replace("non-hispanic Asian", "Non-Hispanic Asian"))
                           .apply(lambda x: x.replace("non-hispanic aian", "Non-Hispanic AIAN"))
                           .apply(lambda x: x.replace("aian", "AIAN"))
                           .apply(lambda x: x.replace("hispanic", "Hispanic"))
                           .apply(lambda x: x.strip())
                           )
    print("df_filtered[['label']].head(100): ", df_filtered[['label']].tail(200))
    # df_filtered['label'] = df_filtered['label'].apply(lambda x: x.replace("Ldl","LDL").replace(" a1c "," A1C ")
    #                                                   .replace("+instructional+w","+w").replace("(not employed)","- not employed")
    #                                                   .replace("Hdl", "HDL")
    #                                                   )

    my_dpi =200
    for i in range(0, df_filtered.shape[0], 51):
        import matplotlib as mpl

        mpl.rcParams['font.family'] = 'Arial'
        sns.set(font="Arial")

        partial_df = df_filtered.iloc[i:i + 50].copy()
        print("partial_df: ", partial_df.shape)
        plt.figure(figsize=(900 / my_dpi, (2000 / my_dpi) * ((partial_df.shape[0] + 10) / (51 + 10))), dpi=my_dpi) ### size
        # sns.set(style="ticks")
        sns.set_style("darkgrid", {"axes.facecolor": ".9"})
        # sns.set_context("paper")
        ax = sns.barplot(y='label', x=class_name, data=partial_df, hue='color_label',
                         dodge=False, hue_order=hue_order,
                         palette=palette)  # ,palette=[palette[i] for i in partial_df.color.astype(int)])
        # plt.xticks(fontsize=8,rotation=90)# ax.tick_params(axis='both', which='major', labelsize=10)
        # plt.tight_layout()
        plt.legend(title="Predictors", loc='lower right', prop={'size': 23})
        plt.xlabel('Mean |SHAP| (average impact on model output magnitude)', fontsize=12)
        plt.ylabel('Variables', fontsize=12, rotation=0)
        # plt.show()
        plt.xlim(df_filtered[class_name].min() - 0.0001, df_filtered[class_name].max() * 1.02)
        plt.grid()

        mpl.rcParams['font.family'] = 'Arial'
        sns.set(font="Arial")
        legend = plt.legend(loc='lower right', prop={'size': 13})
        frame = legend.get_frame()
        frame.set_facecolor('white')
        # plt.axhline(y=14.5, color='r', linestyle='-')
        ax.yaxis.set_label_coords(-1.1, 1.02)
        # ax.set_ylabel() # position=(x, y)
        # ax.tick_params(axis='y', rotation=90)

        print("write ######################")
        print("Fig\\" +shap_reason+"-"+class_name+ "-Abs-" +str(take_abs) +"-"+ str(i) + '.svg',"\n\n\n")
        plt.subplots_adjust(left=0.01, right=0.9, top=0.9, bottom=0.1)  # right=0.9, top=0.9, bottom=0.1
        # plt.show()
        plt.savefig( "Fig\\" +shap_reason+"-"+class_name+ "-Abs-" +str(take_abs) +"-"+ str(i) + '.svg', bbox_inches="tight",
                    pad_inches=0.3, format='svg')  # facecolor='y', , transparent=True, dpi=200 , format='eps'
        # plt.savefig(dataLocation + "Figs/" + "Abs-" + str(i), bbox_inches="tight",
        #             pad_inches=0.3)
        plt.clf()




