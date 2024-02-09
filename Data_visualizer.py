import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import re

included = "SEX_A-1"
outcome = ""
# outcome = "High_impact_chronic_pain"
for outcome in ["Chronic_Pain","High_impact_chronic_pain"]:
    # sub_folder = 'shap1'
    sub_folder ="shapRes-Chronic_Pain-"+included
    if outcome == "High_impact_chronic_pain":
        # sub_folder = 'shap2'
        sub_folder = "shapRes-High_impact_chronic_pain-" + included
    print("outcome: ",outcome,sub_folder)

    variable_list_df = pd.read_excel('NHIS variable list_Modified.xlsx')
    # variable_list_df = variable_list_df[~variable_list_df['category'].isin(['nan', 'filter',np.nan])] # drop filter
    variable_list_df['category'] = variable_list_df['category'].apply(lambda x: x.title() if isinstance(x, str) else x)
    # variable_list_df['category'] = variable_list_df['category'].apply(str.title)
    selected_columns = variable_list_df['variable(s)'].tolist()
    # description # category
    column_desc = dict(zip(variable_list_df['variable(s)'].str.upper(),variable_list_df['description']))
    column_cat = dict(zip(variable_list_df['variable(s)'].str.upper(), variable_list_df['category']))
    print("column_desc ",len(column_desc), column_desc)
    print("column_cat ",len(column_cat), column_cat)
    print("cats:", set(variable_list_df['category']))
    feature_names = pd.read_csv('./'+sub_folder+'/columns.csv')['Column Names'].values
    print("feature_names ", len(feature_names), feature_names )

    arr_shape = np.loadtxt('./'+sub_folder+'/shape.csv')
    shap_values = [ [] for i in range(int(arr_shape))]
    print("load")
    for i in range(int(arr_shape)):
        shap_values[i] = np.loadtxt('./'+sub_folder+'/shap_'+str(i)+".csv")
    print("loaded")
    print('\nD1 Classes:',len(shap_values),'\nD2 samples:', len(shap_values[0]))#,'\nD3 Columns/features:',len(shap_values[0][0]),'\nvalue:',shap_values[0][0][0])
    print('type: ',type(shap_values))
    print('type [0]: ', type(shap_values[0]))
    average_shap_values = np.mean(np.abs(shap_values), axis=0)
    print("average_shap_values shape", average_shap_values.shape)

    df = pd.DataFrame(average_shap_values, columns=['values'],index=feature_names)
    print(df.head(),"\n",str(list(df.index)),"\n", len(list(df.index)),"\n")
    df[['label','cat','color']] = np.nan
    df["inx"]=df.index
    df['label'] = df["inx"].str.split('__', expand=True).apply(lambda x: f"{column_desc.get(x[0], '')} [{(str(x[0]).capitalize())}] ({(str(x[1]).capitalize())})", axis=1)#[0].map(column_desc)
    # df['label'] = df['label'].str.replace('(None)', '')
    df['cat'] = df["inx"].str.split('__', expand=True)[0].map(column_cat)
    df = df[~df['cat'].isin(['nan', 'Filter', np.nan])]  # drop filter
    df = df[~df['cat'].isin(['nan', 'Filter', np.nan])]
    df = df[~df["inx"].isin(['MARITAL_A__9'])]
    print("set(df['cat'].unique())", set(df['cat'].unique()))
    df['color'] = df.cat.map({'risk factor':1, 'covariate':2, 'filter':3, 'risk factor and moderator':4, 'SES':5 })
    df['color_label'] = df['cat']
    df = df.sort_values(by='values', ascending=False, )
    # df = df[~df['category'].isin(['nan', 'filter'])] # drop filter
    print(df.head(75),"\n")
    # df['label'] = df['label'].str.replace("  "," ").replace("\n"," ").replace("\t"," ")

    palette = sns.color_palette("bright", 10) # pastel
    palette = {"Geographic": palette[1], "Socioeconomic Position": palette[3], "Primary Outcome": palette[0],
               "Demographic": palette[9] , 'Physical Health': palette[4], 'Mental Health': palette[2]}  # 7 grey 5 dark red
    hue_order = ['Geographic', 'Socioeconomic Position', 'Primary Outcome', 'Demographic', 'Physical Health', 'Mental Health']

    df_filtered = df.copy()
    df_filtered['index_df'] = df.index
    df_filtered['values'] = df_filtered['values'].abs()
    df_filtered = df_filtered.sort_values(by='values', ascending=False).reset_index(drop=True)
    df_filtered['label']= df_filtered['label'].apply(lambda x: x.capitalize() if isinstance(x, str) else x)
    df_filtered['label']= (df_filtered['label']
                           .apply(lambda x: x.replace(" (none)","")) #.replace(r"\(none\)", "", regex=True)
                           .replace(r"nh ", "non-hispanic ", regex=True) #.replace("nh ", "Non-Hispanic")
                           .apply(lambda x: re.sub(r'\[.*?\]', '', x))
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
                           .apply(lambda x: x.replace("non-hispanic aian", "Non-Hispanic Aian"))
                           .apply(lambda x: x.replace("aian", "Aian"))
                           .apply(lambda x: x.replace("hispanic", "Hispanic"))
                           .apply(lambda x: x.strip())
                           )
    print("df_filtered[['label']].head(): ", df_filtered[['label']].head())
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
        ax = sns.barplot(y='label', x="values", data=partial_df, hue='color_label',
                         dodge=False, hue_order=hue_order,
                         palette=palette)  # ,palette=[palette[i] for i in partial_df.color.astype(int)])
        # plt.xticks(fontsize=8,rotation=90)# ax.tick_params(axis='both', which='major', labelsize=10)
        # plt.tight_layout()
        plt.legend(title="Predictors", loc='lower right', prop={'size': 23})
        plt.xlabel('Mean |SHAP| (average impact on model output magnitude)', fontsize=12)
        plt.ylabel('Variables', fontsize=12, rotation=0)
        # plt.show()
        plt.xlim(df_filtered['values'].min() - 0.0001, df_filtered['values'].max() * 1.02)
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
        print("Fig\\" +outcome+"-"+included+ "-Abs-" + str(i) + '.svg',"\n\n\n")
        plt.subplots_adjust(left=0.01, right=0.9, top=0.9, bottom=0.1)  # right=0.9, top=0.9, bottom=0.1
        # plt.show()
        plt.savefig( "Fig\\" +outcome+"-"+included+ "-Abs-" + str(i) + '.svg', bbox_inches="tight",
                    pad_inches=0.3, format='svg')  # facecolor='y', , transparent=True, dpi=200 , format='eps'
        # plt.savefig(dataLocation + "Figs/" + "Abs-" + str(i), bbox_inches="tight",
        #             pad_inches=0.3)
        plt.clf()

    # exit()


