#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  7 23:09:19 2019

@author: yujzhang
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  7 16:38:24 2019

@author: yujzhang
"""

import pandas as pd
import glob
import sklearn.utils as su
import sklearn.ensemble as se
import sklearn.metrics as sm
import matplotlib.pyplot as mp
import numpy as np
import sklearn.tree as st
import seaborn as sns
import matplotlib.ticker as ticker
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import log_loss

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

import plotly.figure_factory as ff


path = '.'
all_files = glob.glob(path + "/*.csv")
li = []

for filename in all_files:
    df = pd.read_csv(filename, index_col=None, header=0)
    li.append(df)

frame = pd.concat(li, axis=0, ignore_index=True)


#for file in files:
#    df = pd.read_csv(file, index_col=None, header=0)


#Continus Variables
age = frame['age']
education_num = frame['education-num']
capital_gain = frame['capital-gain']
capital_loss = frame['capital-loss']
hours_per_week = frame['hours-per-week']

#Category Variables
workclass = frame['workclass']
marital_status = frame['marital-status']
occupation = frame['occupation']
relationship = frame['relationship']
race = frame['race']
sex = frame['sex']
native_country = frame['native-country']
income = frame['income']

column_list = ['age', 'education-num', 'hours-per-week']

cat_col_list = ['workclass', 'education', 'marital-status', 'occupation', 
                'relationship', 'race', 'sex','native-country']

col_list = frame.columns
print(col_list)
#=============================Check missing values==========================
missing_values = frame.isna()

attr = []
percentage = []

for coll in missing_values.columns:
    print(coll)
    print(np.mean(missing_values[coll]))
    attr.append(coll)
    percentage.append(1-np.mean(missing_values[coll]))
    
result1 = pd.DataFrame(columns=['attributes','non_missing_value_percentage'])
result1['attributes'] = attr
result1['non_missing_value_percentage'] = percentage 

f, (g) = mp.subplots(figsize=(18, 6))
g=sns.barplot(x='attributes',y='non_missing_value_percentage',data=result1)

for index, row in result1.iterrows():
    g.text(row.name,row.non_missing_value_percentage, 
           round(row.non_missing_value_percentage,2), color='black', ha="center")

mp.show()
    
#=================Check zero values======================================

y = [] 
x = [] 
z = [] 
 

for cl in column_list:
    print(cl + ":")
    print(np.mean(frame[cl]==0))
    y.append(np.mean(frame[cl]==0))
    x.append(cl)


result = pd.DataFrame(columns=['attributes','zero_value_percentage'])
result['attributes'] = x
result['zero_value_percentage'] = y  

f, (g) = mp.subplots(figsize=(10, 10))
g=sns.barplot(x='attributes',y='zero_value_percentage',data=result, palette="ch:.25")

for index, row in result.iterrows():
    g.text(row.name,row.zero_value_percentage, round(row.zero_value_percentage,2), color='black', ha="center")

mp.show()



#=====================Clean Data==========================================

clean_data = frame.drop(columns=['capital-gain', 'capital-loss'])

clean_data1 = clean_data.dropna()

for x in clean_data.columns:
    mv = clean_data[x].isin(["?"]).sum()
    if mv > 0:
        clean_data = clean_data[clean_data[x] != '?']
        



#=======================Visualization=====================================


df1 = clean_data1[['marital-status','education-num','income']]

df2 = clean_data1[['workclass','education-num','income']]

print(df1.head())

heatmap1_data = pd.pivot_table(df1, values='education-num', 
                     index=['income'], 
                     columns='marital-status')

sns.heatmap(heatmap1_data, cmap="YlGnBu")

#========================Describe the clean data=====================
print(clean_data1.describe())

#random_data = np.random.randn(50000,2)  * 20 + 20
#print(random_data)


#lb_make = LabelEncoder()
#clean_data1["income_code"] = lb_make.fit_transform(clean_data1["income"])

#data_f = pd.DataFrame(clean_data1['age'],clean_data1['income'])
#print(data_f)
f, (ax1) = mp.subplots(figsize=(10, 10))
ax1 = sns.boxplot(x = 'income', y = 'age', data=clean_data1, showfliers=True)

f, (ax2) = mp.subplots(figsize=(10, 10))
ax2 = sns.boxplot(x = 'income', y = 'education-num', data=clean_data1, showfliers=True)

f, (ax3) = mp.subplots(figsize=(10, 10))
ax3 = sns.boxplot(x = 'income', y = 'hours-per-week', data=clean_data1, showfliers=True)



#========================Data Transformation=====================
label_encoders = {}

processed_data = pd.DataFrame()
for col in cat_col_list:
    print("Encoding {}".format(col))
    new_le = LabelEncoder()
    processed_data[col] = new_le.fit_transform(clean_data1[col])
    label_encoders[col] = new_le

  
processed_data['age'] = clean_data1['age']
processed_data['education-num'] = clean_data1['education-num']
processed_data['hours-per-week'] = clean_data1['hours-per-week']

new_le = LabelEncoder()
processed_data['income'] = new_le.fit_transform(clean_data1['income'])

processed_data = processed_data.dropna()


#========================Fit the models=====================
    
fn_dy = processed_data.columns[1:10]
x = processed_data.iloc[:,1:10]
y = processed_data['income']
  
su.shuffle(x, y, random_state=7)  
train_size = int(len(x) * 0.7)
train_x, test_x, train_y, test_y = x[:train_size], x[train_size:], y[:train_size], y[train_size:]

#RandomForest
model = se.RandomForestClassifier(n_estimators=50, max_depth=6,
                             random_state=5)
model.fit(train_x, train_y) 
print(model)
fi_dy = model.feature_importances_
print(fi_dy)

#KNN
neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(train_x, train_y) 
print(neigh)

#SVM
clf = svm.SVC(gamma='scale',probability=True)
clf.fit(train_x, train_y) 
print(clf)

#Gaussian Classifier
model_G = GaussianNB()
model_G.fit(train_x,train_y)
print(model_G)

#DecisionTree
dtc = DecisionTreeClassifier()
dtc.fit(train_x,train_y)
print(dtc)

#LinearDiscriminantAnalysis
lda = LinearDiscriminantAnalysis()
lda.fit(train_x,train_y)
print(lda)

#Logsitc Regression
logmodel = LogisticRegression()
logmodel.fit(train_x,train_y)
print(logmodel)
    
#feature importance
fi_dy = model.feature_importances_
print(fi_dy)
    
    
#Test if the model is fitting well
pred_test_y = model.predict(test_x)
neigh_pred_test_y = neigh.predict(test_x)
svm_pred_test_y = clf.predict(test_x)
gaussian_predicted_test_y = model_G.predict(test_x)
dtc_y_pred = dtc.predict(test_x)
lda_y_pred = lda.predict(test_x)
log_y_pred = logmodel.predict(test_x)

model_list = [model, neigh, clf, model_G, dtc, lda, logmodel]
alghrithm_list = ["RandomForest", "KNN", "SVM", "Gaussian", "DecisionTree", 
                  "LDA", "Logsitc Regression"]
i = 0
for ml in model_list:
    probs = ml.predict_proba(test_x)[:, 1]
    score = log_loss(test_y, probs)
    print("The Logarithmic Loss of " + alghrithm_list[i] + " is : " + str(score))
    i += 1


probs1 = model.predict_proba(test_x)
probs1 = probs1[:, 1]
score1 = log_loss(test_y, probs1)
print('Logarithmis Loss of RandomForest: %.2f' % score1)

probs2 = neigh.predict_proba(test_x)
probs2 = probs2[:, 1]
score2 = log_loss(test_y, probs2)
print('Logarithmis Loss of RandomForest: %.2f' % score2)

probs3 = clf.predict_proba(test_x)
probs3 = probs3[:, 1]
score3 = log_loss(test_y, probs3)
print('Logarithmis Loss of RandomForest: %.2f' % score3)

probs4 = model_G.predict_proba(test_x)
probs4 = probs4[:, 1]
score4 = log_loss(test_y, probs4)
print('Logarithmis Loss of RandomForest: %.2f' % score4)

probs5 = dtc.predict_proba(test_x)
probs5 = probs5[:, 1]
score5 = log_loss(test_y, probs5)
print('Logarithmis Loss of RandomForest: %.2f' % score5)

probs6 = lda.predict_proba(test_x)
probs6 = probs6[:, 1]
score6 = log_loss(test_y, probs6)
print('Logarithmis Loss of RandomForest: %.2f' % score6)

probs7 = logmodel.predict_proba(test_x)
probs7 = probs7[:, 1]
score7 = log_loss(test_y, probs7)
print('Logarithmis Loss of RandomForest: %.2f' % score7)


print("======================================")
print(pred_test_y)
print("======================================")
target_names = ['class 0', 'class 1']

pred_y_list = [pred_test_y,neigh_pred_test_y,svm_pred_test_y,
               gaussian_predicted_test_y,dtc_y_pred,lda_y_pred,log_y_pred]

count = 0

for pred_y in pred_y_list:
    print("The classification report of " + alghrithm_list[count])
    print(classification_report(test_y, pred_y, target_names=target_names))
    count += 1


count2 = 0
for pre_y in pred_y_list:
    df = pd.DataFrame(test_y,pre_y)
    confusion_matrix = pd.crosstab(test_y,pre_y, rownames=['Actual'], 
                                   colnames=['Predicted'])
    #print(confusion_matrix)
    mp.figure(figsize=(10,10))
    mp.title('Confusion Matrix of '+ alghrithm_list[count2])
    sns.heatmap(confusion_matrix, annot=True, fmt="d")
    count2 += 1
    
    
mp.figure(figsize=(10,10))
sns.heatmap(processed_data.corr(), annot=True,fmt=".2f")


mp.figure('Random Forest')
mp.subplot(211)
mp.title('Random Forest', fontsize=16)
mp.ylabel('Importance', fontsize=12)
mp.tick_params(labelsize=10)
mp.grid(axis='y', linestyle=':')
sorted_indices = fi_dy.argsort()[::-1]
pos = np.arange(sorted_indices.size)
mp.bar(pos, fi_dy[sorted_indices])
mp.xticks(pos, processed_data.columns[sorted_indices],
          rotation=90)
mp.tight_layout()
mp.show()


importance = pd.DataFrame()
importance['importance'] = fi_dy[sorted_indices]
importance['feature'] = processed_data.columns[sorted_indices]

print(importance)
mp.figure('Random Forest')
sns.barplot('feature', y='importance', data=importance, palette="BuGn_r")
mp.xticks(rotation=90)


from sklearn.model_selection import GridSearchCV
grid_values = {'penalty': ['l1', 'l2'],'C':[0.001,.009,0.01,.09,1,5,10,25]}
grid_logmodel_acc = GridSearchCV(logmodel, param_grid = grid_values,scoring = 'recall')
grid_logmodel_acc.fit(train_x, train_y)

#Predict values based on new parameters
y_pred_acc = grid_logmodel_acc.predict(test_x)
print(classification_report(test_y, y_pred_acc, target_names=target_names))


rfc=se.RandomForestClassifier(random_state=42)
param_grid = { 
    'n_estimators': [100],
    'max_features': ['auto'],
    'max_depth' : [6],
    'criterion' :['gini']
}

CV_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv= 5)
CV_rfc.fit(train_x, train_y)
print(CV_rfc.best_params_)


clf_gs = svm.SVC(gamma='scale',probability=True)
parameters = {'kernel':['linear', 'rbf','poly'], 
              'gamma': [1,2,3,'auto'],
              'decision_function_shape':['ovo','ovr'],
              'shrinking':[True,False]}
clf_gs = GridSearchCV(clf_gs, parameters)
clf_gs.fit(train_x,train_y)
print(clf_gs.best_params_)

lr = LogisticRegression()
grid_values = {'penalty': ['l1', 'l2'],'C':[0.001,.009,0.01,.09,1,5,10,25]}
grid_clf_acc = GridSearchCV(lr, param_grid = grid_values,scoring = 'recall')
grid_clf_acc.fit(train_x, train_y)
print(clf.best_params_)

clf_f = svm.SVC(probability=True,kernel='linear',gamma=3, 
                decision_function_shape='ovo',shrinking=True)
clf_f.fit(train_x, train_y) 
pred_y_clf_f = clf_f.predict(test_x)
print(classification_report(test_y, pred_y_clf_f, target_names=target_names))

clf_f1 = svm.SVC(probability=True,kernel='poly',gamma=4, 
                decision_function_shape='ovo',shrinking=True)
clf_f1.fit(train_x, train_y) 
pred_y_clf_f1 = clf_f1.predict(test_x)
print(classification_report(test_y, pred_y_clf_f1, target_names=target_names))


#Logsitc Regression
logmodel_n = LogisticRegression(C=1, penalty='l1')
logmodel_n.fit(train_x,train_y)
pred_y_lr = logmodel_n.predict(test_x)
print(classification_report(test_y, pred_y_lr, target_names=target_names))

clf_n = svm.SVC(gamma=3,probability=True,kernel='rbf',
                decision_function_shape='ovr',shrinking=True)
clf_n.fit(train_x, train_y) 
pred_y_svm = clf_n.predict(test_x)
print(classification_report(test_y, pred_y_svm, target_names=target_names))

#RandomForest
model_rf = se.RandomForestClassifier(n_estimators=100, max_depth=6,
                             random_state=42, max_features='auto')
model_rf.fit(train_x, train_y) 
pred_y_rf = model_rf.predict(test_x)
print(classification_report(test_y, pred_y_rf, target_names=target_names))