#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 29 21:50:12 2019

@author: yujzhang
"""

for position in positions:
    
    group_data = clean_data[clean_data['Position'] == position]  
    
    print(group_data.columns)
    
    
    index = [7,8,9,11,12,13,15,20,21,22,23,24,25,26]
    
    fn_dy = group_data.columns[index]
    x = group_data.iloc[:,index]
    y = group_data['Rank']
    
    su.shuffle(x, y, random_state=7)
    
    
    train_size = int(len(x) * 0.9)
    train_x, test_x, train_y, test_y = x[:train_size], x[train_size:], y[:train_size], y[train_size:]
    
    #RandomForestRegressor
    #model = se.RandomForestRegressor(max_depth=10, n_estimators=1000, min_samples_split=2)
    #model.fit(train_x, train_y)
    #feature importance
    #fi_dy = model.feature_importances_
    #Test if the model is fitting well
    #pred_test_y = model.predict(test_x)
    #print("The r2 score of RandomForestRegresson is :")
    #print(sm.r2_score(test_y, pred_test_y))
    
    #RandomForest
    model = se.RandomForestClassifier(n_estimators=50, max_depth=6,
                             random_state=5)
    model.fit(train_x, train_y) 
    print(model)
    fi_dy = model.feature_importances_
    print(fi_dy)
    pred_test_y = model.predict(test_x)
    print("The importance of decisicion tree classifier is : ")
    print(fi_dy)
    print("The F1 score of decision tree classifier is : ")
    print(f1_score(test_y, pred_test_y, average=None))
    
    
    #DecisionTreeRegressor                                 
   # model2 = st.DecisionTreeRegressor(max_depth=4)
    #Train
   # model2.fit(train_x,train_y)
   # fi_dt = model2.feature_importances_
    #Predict
   # pred_test_y2 = model2.predict(test_x)
   # print("The r2 score of DecisionTreeRegression is :") 
   # print(sm.r2_score(test_y, pred_test_y2)) 
    
    
    dtc = DecisionTreeClassifier()
    dtc.fit(train_x,train_y)
    fi_dtc = dtc.feature_importances_
    pred_test_y2_dtc = dtc.predict(test_x)
    print("The importance of decisicion tree classifier is : ")
    print(fi_dtc)
    print("The F1 score of decision tree classifier is : ")
    print(f1_score(test_y, pred_test_y2_dtc, average=None))
    
    
    #AdaBoostRegressor
    #model3 = se.AdaBoostRegressor(
    #        st.DecisionTreeRegressor(max_depth=4),
    #        n_estimators=400,random_state=7)
    #model3.fit(train_x,train_y)
    #fi_ab = model3.feature_importances_
    #pred_test_y3 = model3.predict(test_x)
    #print("The r2 score of AdaBoostRegression is :")
    #print(sm.r2_score(test_y, pred_test_y3))   
    
    
    abc = AdaBoostClassifier(n_estimators=50,
                         learning_rate=1)
    model3 = abc.fit(train_x, train_y)
    fi_abc = model3.feature_importances_
    y_pred_abc = model.predict(test_x)    
    print("The importance of decisicion tree classifier is : ")
    print(fi_abc)
    print("The F1 score of decision tree classifier is : ")
    print(f1_score(test_y, y_pred_abc, average=None))
    
    mp.figure('Feature Importance', figsize=(18, 12.5))
    mp.subplot(131)
    mp.title('Random Forest', fontsize=16)
    mp.ylabel('Importance', fontsize=12)
    mp.tick_params(labelsize=10)
    mp.grid(axis='y', linestyle=':')
    sorted_indices = fi_dy.argsort()[::-1]
    pos = np.arange(sorted_indices.size)
    mp.bar(pos, fi_dy[sorted_indices], facecolor='deepskyblue', edgecolor='steelblue')
    mp.xticks(pos, fn_dy[sorted_indices],
              rotation=90)
    mp.tight_layout()
    
    mp.subplot(132)
    mp.title('Decision Tree', fontsize=16)
    mp.ylabel('Importance', fontsize=12)
    mp.tick_params(labelsize=10)
    mp.grid(axis='y', linestyle=':')
    sorted_indices = fi_dtc.argsort()[::-1]
    pos = np.arange(sorted_indices.size)
    mp.bar(pos, fi_dtc[sorted_indices], facecolor='deepskyblue', edgecolor='steelblue')
    mp.xticks(pos, fn_dy[sorted_indices],
          rotation=90)
    mp.tight_layout()
    
    
    mp.subplot(133)
    mp.title('AdaBoost Decision Tree', fontsize=16)
    mp.ylabel('Importance', fontsize=12)
    mp.tick_params(labelsize=10)
    mp.grid(axis='y', linestyle=':')
    sorted_indices = fi_abc.argsort()[::-1]
    pos = np.arange(sorted_indices.size)
    mp.bar(pos, fi_abc[sorted_indices], facecolor='indianred', edgecolor='indianred')
    mp.xticks(pos, fn_dy[sorted_indices],
              rotation=90)
    mp.tight_layout()
    
    mp.savefig('/Users/yujzhang/Downloads/backup/plot'+position+'.png', format='png')
    mp.close()
