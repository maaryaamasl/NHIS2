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
import h2o
from h2o.automl import H2OAutoML
# import autokeras as ak
# from autokeras import StructuredDataClassifier
import shap
# import shapley
from xgboost import XGBClassifier
# from lightgbm import LGBMClassifier



print("\ncleaned_data")
cleaned_data = pd.read_csv('Cleaned_data_2019.csv')

print('cleaned_data: ',cleaned_data.shape)
# Chronic_Pain {0, 1}
# High_impact_chronic_pain {0, 1}
outcomes = ['Chronic_Pain', 'High_impact_chronic_pain']
for column in outcomes:
    print(column, set(cleaned_data[column]), cleaned_data[column].value_counts().values)
# Outcome <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< VARIABLES & OUTCOMES
print("######### Setting ########")
outcome = ['High_impact_chronic_pain'] # 'Chronic_Pain', 'High_impact_chronic_pain'
filtering="HISPALLP_A__NH Black/African-American" # "HISPALLP_A__NH White" # SEX_A
val = 1
shap_reason = "shapRes-High_impact_chronic_pain-HISPALLP_A__NH Black-African-American-1"
print(shap_reason,outcome,filtering,val)
print("######### Filter ###########")
print('cleaned_data: ',cleaned_data.shape)
cleaned_data = cleaned_data[(cleaned_data[filtering] == val)] # & (selected_data['PAIWKLM3M_A'] == 1)
cleaned_data.drop([filtering], axis=1, inplace=True)
print('cleaned_data: ',cleaned_data.shape)

drop_col = [x for x in outcomes if x not in outcome]
print("Outcome:",outcome," \nDropped_col:",drop_col)
cleaned_data.drop(drop_col, axis=1, inplace=True) # 'High_impact_chronic_pain'
for column in cleaned_data.columns:
    if filtering in column:
        print(column, set(cleaned_data[column]))



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
X = cleaned_data.drop(outcome, axis=1)  # Features
Y = cleaned_data[outcome]  # Target
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

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



print("XGboost") ############################### <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< Chosen model
clf = XGBClassifier()
clf.fit(X, Y)
y_pred_clf = clf.predict(X)
accuracy = accuracy_score(Y, y_pred_clf)
auc = roc_auc_score(Y, y_pred_clf)
print("Accuracy (XGBoost Classifier):",outcome, accuracy, auc)
# ['High_impact_chronic_pain'] 0.9716733806509368 0.8425071151403117
# ['High_impact_chronic_pain'] 0.9234335038363172 0.6440411875664475
# ['Chronic_Pain'] 0.8935673636421766 0.7915005806795882
# ['Chronic_Pain'] 0.835485933503836 0.66474041732368
# exit(-1)

def custom_predict(X):
    return clf.predict(X)
kmeans_k =100 # 100
rows_devideby_to_use = 1 # 1
explainer = shap.KernelExplainer(custom_predict, shap.kmeans(X.values, kmeans_k))
number_of_rows = X.values.shape[0]
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
    np.savetxt("./"+shap_reason+"/shap_"+str(i)+".csv", shap_values[i])
np.savetxt("./"+shap_reason+"/shape.csv",np.array([len(shap_values)]))

column_names = X.columns.values
print(column_names)
pd.DataFrame(column_names, columns=['Column Names']).to_csv("./"+shap_reason+'/columns.csv', index=False)

exit()

print("h2o")  ############################### <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
h2o.init(max_mem_size="8G")
aml = H2OAutoML(max_models=15, seed=1, sort_metric = "accuracy") # before 20 eresult below
x=X.columns.tolist()
y=Y.columns.tolist()[0]
cleaned_data_h2o= h2o.H2OFrame(cleaned_data)
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
exit(1)
# deeplearning prediction progress: |██████████████████████████████████████████████| (done) 100%
# Accuracy of the leader model: [[0.5534361954245438, 0.9229146381045116]]
# model_id                                                accuracy       auc    logloss     aucpr    mean_per_class_error      rmse        mse
# DeepLearning_grid_2_AutoML_1_20240121_222748_model_1    0.92252   0.847557   0.230072  0.431442                0.287611  0.247761  0.0613856
# GBM_4_AutoML_1_20240121_222748                          0.922679  0.875458   0.206759  0.458941                0.273871  0.243211  0.0591517
# GBM_grid_1_AutoML_1_20240121_222748_model_1             0.922719  0.869304   0.20902   0.460876                0.252873  0.243135  0.0591149
# GBM_3_AutoML_1_20240121_222748                          0.923435  0.879233   0.204128  0.464329                0.267852  0.242019  0.0585732
# XRT_1_AutoML_1_20240121_222748                          0.923475  0.870231   0.214913  0.461044                0.27233   0.244795  0.0599248
# DeepLearning_grid_3_AutoML_1_20240121_222748_model_1    0.924112  0.84898    0.236742  0.432182                0.288286  0.247011  0.0610144
# GBM_5_AutoML_1_20240121_222748                          0.92463   0.883996   0.200632  0.472756                0.246039  0.239877  0.0575408
# GBM_grid_1_AutoML_1_20240121_222748_model_3             0.924988  0.88471    0.199138  0.486868                0.266849  0.238852  0.0570503
# DeepLearning_1_AutoML_1_20240121_222748                 0.924988  0.872187   0.206517  0.463083                0.267588  0.241924  0.0585271
# DeepLearning_grid_1_AutoML_1_20240121_222748_model_1    0.925147  0.860048   0.224206  0.466974                0.264188  0.243806  0.0594414
# [17 rows x 8 columns]
#
#                                              model_id  accuracy       auc   logloss     aucpr  mean_per_class_error      rmse       mse
# 0   DeepLearning_grid_2_AutoML_1_20240121_222748_m...  0.922520  0.847557  0.230072  0.431442              0.287611  0.247761  0.061386
# 1                      GBM_4_AutoML_1_20240121_222748  0.922679  0.875458  0.206759  0.458941              0.273871  0.243211  0.059152
# 2         GBM_grid_1_AutoML_1_20240121_222748_model_1  0.922719  0.869304  0.209020  0.460876              0.252873  0.243135  0.059115
# 3                      GBM_3_AutoML_1_20240121_222748  0.923435  0.879233  0.204128  0.464329              0.267852  0.242019  0.058573
# 4                      XRT_1_AutoML_1_20240121_222748  0.923475  0.870231  0.214913  0.461044              0.272330  0.244795  0.059925
# 5   DeepLearning_grid_3_AutoML_1_20240121_222748_m...  0.924112  0.848980  0.236742  0.432182              0.288286  0.247011  0.061014
# 6                      GBM_5_AutoML_1_20240121_222748  0.924630  0.883996  0.200632  0.472756              0.246039  0.239877  0.057541
# 7         GBM_grid_1_AutoML_1_20240121_222748_model_3  0.924988  0.884710  0.199138  0.486868              0.266849  0.238852  0.057050
# 8             DeepLearning_1_AutoML_1_20240121_222748  0.924988  0.872187  0.206517  0.463083              0.267588  0.241924  0.058527
# 9   DeepLearning_grid_1_AutoML_1_20240121_222748_m...  0.925147  0.860048  0.224206  0.466974              0.264188  0.243806  0.059441
# 10                     GBM_2_AutoML_1_20240121_222748  0.925386  0.882719  0.200962  0.481337              0.252478  0.240018  0.057609
# 11                     DRF_1_AutoML_1_20240121_222748  0.925705  0.871728  0.219204  0.473004              0.268274  0.240948  0.058056
# 12                     GBM_1_AutoML_1_20240121_222748  0.926541  0.883398  0.197398  0.496299              0.236961  0.237214  0.056271
# 13        GBM_grid_1_AutoML_1_20240121_222748_model_2  0.926700  0.886618  0.196753  0.502157              0.253042  0.236947  0.056144
# 14                     GLM_1_AutoML_1_20240121_222748  0.927735  0.885340  0.196150  0.507843              0.250792  0.236451  0.055909
# 15  StackedEnsemble_BestOfFamily_1_AutoML_1_202401...  0.927974  0.887538  0.194858  0.509150              0.254042  0.235900  0.055649
# 16  StackedEnsemble_AllModels_1_AutoML_1_20240121_...  0.928054  0.888472  0.194085  0.513450              0.244769  0.235375  0.055401
# gbm prediction progress: |███████████████████████████████████████████████████████| (done) 100%
# Accuracy of the leader model: [[0.5528316012672295, 0.8171048360921779]]
# model_id                                                accuracy       auc    logloss     aucpr    mean_per_class_error      rmse       mse
# GBM_grid_1_AutoML_1_20240121_234233_model_1             0.813067  0.803505   0.423637  0.58842                 0.274527  0.366786  0.134532
# DeepLearning_grid_2_AutoML_1_20240121_234233_model_1    0.813704  0.800252   0.436212  0.575839                0.279695  0.370381  0.137182
# XRT_1_AutoML_1_20240121_234233                          0.814142  0.804537   0.437443  0.585215                0.276449  0.371723  0.138178
# GBM_4_AutoML_1_20240121_234233                          0.814779  0.812285   0.417084  0.598801                0.271514  0.363952  0.132461
# DeepLearning_grid_1_AutoML_1_20240121_234233_model_1    0.815616  0.797497   0.448052  0.587387                0.28119   0.371022  0.137657
# DRF_1_AutoML_1_20240121_234233                          0.815775  0.80789    0.420336  0.595407                0.270397  0.364885  0.133141
# DeepLearning_1_AutoML_1_20240121_234233                 0.816054  0.812244   0.41831   0.592493                0.268618  0.363852  0.132389
# DeepLearning_grid_3_AutoML_1_20240121_234233_model_1    0.816173  0.803234   0.42605   0.593057                0.279616  0.366476  0.134305
# GBM_3_AutoML_1_20240121_234233                          0.818004  0.817945   0.411659  0.608575                0.270035  0.36125   0.130502
# GBM_2_AutoML_1_20240121_234233                          0.818403  0.81914    0.410952  0.607236                0.263304  0.360937  0.130275
# [17 rows x 8 columns]
#
#                                              model_id  accuracy       auc   logloss     aucpr  mean_per_class_error      rmse       mse
# 0         GBM_grid_1_AutoML_1_20240121_234233_model_1  0.813067  0.803505  0.423637  0.588420              0.274527  0.366786  0.134532
# 1   DeepLearning_grid_2_AutoML_1_20240121_234233_m...  0.813704  0.800252  0.436212  0.575839              0.279695  0.370381  0.137182
# 2                      XRT_1_AutoML_1_20240121_234233  0.814142  0.804537  0.437443  0.585215              0.276449  0.371723  0.138178
# 3                      GBM_4_AutoML_1_20240121_234233  0.814779  0.812285  0.417084  0.598801              0.271514  0.363952  0.132461
# 4   DeepLearning_grid_1_AutoML_1_20240121_234233_m...  0.815616  0.797497  0.448052  0.587387              0.281190  0.371022  0.137657
# 5                      DRF_1_AutoML_1_20240121_234233  0.815775  0.807890  0.420336  0.595407              0.270397  0.364885  0.133141
# 6             DeepLearning_1_AutoML_1_20240121_234233  0.816054  0.812244  0.418310  0.592493              0.268618  0.363852  0.132389
# 7   DeepLearning_grid_3_AutoML_1_20240121_234233_m...  0.816173  0.803234  0.426050  0.593057              0.279616  0.366476  0.134305
# 8                      GBM_3_AutoML_1_20240121_234233  0.818004  0.817945  0.411659  0.608575              0.270035  0.361250  0.130502
# 9                      GBM_2_AutoML_1_20240121_234233  0.818403  0.819140  0.410952  0.607236              0.263304  0.360937  0.130275
# 10                     GBM_1_AutoML_1_20240121_234233  0.818482  0.819451  0.408925  0.616591              0.270580  0.359757  0.129425
# 11                     GLM_1_AutoML_1_20240121_234233  0.818880  0.819203  0.410054  0.613116              0.263146  0.360671  0.130083
# 12                     GBM_5_AutoML_1_20240121_234233  0.819040  0.820097  0.409750  0.612616              0.262514  0.360245  0.129776
# 13        GBM_grid_1_AutoML_1_20240121_234233_model_3  0.819358  0.821701  0.408177  0.615903              0.264089  0.359633  0.129336
# 14  StackedEnsemble_BestOfFamily_1_AutoML_1_202401...  0.819796  0.821783  0.407495  0.616476              0.261053  0.359467  0.129216
# 15        GBM_grid_1_AutoML_1_20240121_234233_model_2  0.820154  0.822474  0.407186  0.619776              0.265480  0.358980  0.128867
# 16  StackedEnsemble_AllModels_1_AutoML_1_20240121_...  0.820792  0.824028  0.405217  0.622318              0.261208  0.358173  0.128288
exit (-1) ############################### <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< Old runs

# https://docs.h2o.ai/h2o/latest-stable/h2o-docs/automl.html
print("h2o")  ############################### <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
h2o.init()
aml = H2OAutoML(max_models=3, seed=1) # before 20 eresult below
x=X.columns.tolist()
y=Y.columns.tolist()[0]
cleaned_data_h2o= h2o.H2OFrame(cleaned_data)
cleaned_data_h2o[y] = cleaned_data_h2o[y].asfactor()
print(len(x),x,"\n",y)

# aml.train(x=x, y=y, training_frame=cleaned_data_h2o)
# lb = aml.leaderboard
# lb.head(rows=lb.nrows)
# print("H2o: ",lb.head(rows=lb.nrows))
# best_model = h2o.get_model(lb[0, 'model_id'])
# model_path = best_model.save_mojo("./h2o_best_model_mojo")
# print(model_path) # /Users/macpro/PycharmProjects/NHIS/h2o_best_model_mojo/StackedEnsemble_BestOfFamily_1_AutoML_1_20240103_175628.zip
model_path = "/Users/macpro/PycharmProjects/NHIS/h2o_best_model_mojo/StackedEnsemble_BestOfFamily_1_AutoML_1_20240103_175628.zip"
best_model = h2o.import_mojo(model_path)

data_for_shap =cleaned_data_h2o.as_data_frame()
X_shap = data_for_shap.values # .drop(y, axis=1)
def predict_func(X):
    return best_model.predict(h2o.H2OFrame(X)).as_data_frame().values.flatten()
# 82 %
# model_id                                                      auc    logloss     aucpr    mean_per_class_error      rmse       mse
# StackedEnsemble_AllModels_1_AutoML_5_20240103_165139     0.823065   0.405868  0.622559                0.264175  0.358471  0.128501
# StackedEnsemble_BestOfFamily_1_AutoML_5_20240103_165139  0.822348   0.406775  0.620386                0.262379  0.358969  0.128859
# GBM_grid_1_AutoML_5_20240103_165139_model_2              0.821338   0.408253  0.618297                0.261707  0.359459  0.129211
# GBM_5_AutoML_5_20240103_165139                           0.8203     0.408981  0.616893                0.265317  0.359965  0.129575
# GBM_1_AutoML_5_20240103_165139                           0.820063   0.408395  0.61845                 0.263373  0.359695  0.129381
# GBM_2_AutoML_5_20240103_165139                           0.819374   0.410107  0.613996                0.266462  0.360541  0.12999
# GLM_1_AutoML_5_20240103_165139                           0.817865   0.41117   0.61155                 0.262916  0.361108  0.130399
# GBM_3_AutoML_5_20240103_165139                           0.81701    0.412856  0.606417                0.268459  0.362031  0.131066
# XGBoost_3_AutoML_5_20240103_165139                       0.816137   0.413028  0.608252                0.265806  0.362068  0.131093
# XGBoost_grid_1_AutoML_5_20240103_165139_model_3          0.814341   0.414698  0.606744                0.269712  0.362479  0.131391
# GBM_4_AutoML_5_20240103_165139                           0.813646   0.415417  0.604124                0.269013  0.363234  0.131939
# DRF_1_AutoML_5_20240103_165139                           0.808394   0.420776  0.596667                0.272009  0.36478   0.133065
# DeepLearning_1_AutoML_5_20240103_165139                  0.80735    0.421193  0.59573                 0.276512  0.36539   0.13351
# XGBoost_grid_1_AutoML_5_20240103_165139_model_2          0.805408   0.426518  0.593782                0.273987  0.367283  0.134897
# GBM_grid_1_AutoML_5_20240103_165139_model_1              0.804997   0.423626  0.584946                0.275394  0.367026  0.134708
# DeepLearning_grid_1_AutoML_5_20240103_165139_model_1     0.804793   0.435337  0.598059                0.272375  0.368002  0.135426
# DeepLearning_grid_3_AutoML_5_20240103_165139_model_1     0.803743   0.428389  0.596082                0.278294  0.366906  0.13462
# XRT_1_AutoML_5_20240103_165139                           0.80327    0.436694  0.585898                0.277114  0.371433  0.137962
# XGBoost_grid_1_AutoML_5_20240103_165139_model_1          0.801576   0.431547  0.581897                0.280946  0.370062  0.136946
# DeepLearning_grid_2_AutoML_5_20240103_165139_model_1     0.801469   0.430958  0.585982                0.273034  0.367842  0.135308
# XGBoost_2_AutoML_5_20240103_165139                       0.796801   0.439078  0.576644                0.277189  0.37254   0.138786
# XGBoost_1_AutoML_5_20240103_165139                       0.796613   0.438543  0.578245                0.283792  0.372661  0.138876
exit(1)

# XGboost
print("XGboost") ############################### <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
clf = XGBClassifier()
clf.fit(x_train, y_train)
y_pred_clf = clf.predict(x_test)
accuracy = accuracy_score(y_test, y_pred_clf)
print("Accuracy (XGBoost Classifier):", accuracy)
# clf2 = LGBMClassifier()
# clf2.fit(X_train, y_train)
# y_pred_clf2 = clf2.predict(x_test)
# accuracy2 = accuracy_score(y_test, y_pred_clf2)
# print("Accuracy (light XGBoost Classifier):", accuracy)
# 80%
# Accuracy (XGBoost Classifier): 0.8035485933503836
exit(-1)

# auto Keras
print("auto keras")  ############################### <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
search = StructuredDataClassifier(max_trials=2)
search.fit(x=x_train, y=y_train, verbose=0)
loss, acc = search.evaluate(x_test, y_test, verbose=0)
print('Accuracy auto keras: %.3f' % acc)
# %
# out
exit(-1)

# https://epistasislab.github.io/tpot/api/
print("Tpot")  ############################### <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
model = TPOTClassifier(scoring='accuracy', verbosity=2, random_state=1, n_jobs=-1) # generations=5, population_size=50,
model.fit(x_train, y_train)
model.export('tpot_best_model.py')
# 79%
# Best pipeline: XGBClassifier(CombineDFs(input_matrix, input_matrix), learning_rate=0.1, max_depth=3, min_child_weight=18, n_estimators=100, n_jobs=1, subsample=0.15000000000000002, verbosity=0)
# Best pipeline: XGBClassifier(CombineDFs(input_matrix, CombineDFs(CombineDFs(CombineDFs(GaussianNB(input_matrix), input_matrix), input_matrix), MLPClassifier(input_matrix, alpha=0.001, learning_rate_init=0.1))), learning_rate=0.1, max_depth=3, min_child_weight=1, n_estimators=100, n_jobs=1, subsample=0.5, verbosity=0)
exit(1)

# Random Forest
print("random forest") ############################### <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
clf = RandomForestClassifier(n_estimators=1000,random_state=42)
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
print(f"Accuracy Random forest: {accuracy}")
print("Classification Report:")
print(report)
# 80%
# out
exit(-1)