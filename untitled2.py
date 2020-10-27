#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 29 18:58:18 2019

@author: yujzhang
"""

#AdaBoostRegressor
    model3 = se.AdaBoostRegressor(
            st.DecisionTreeRegressor(max_depth=4),
            n_estimators=400,random_state=7)
    model3.fit(train_x,train_y)
    fi_ab = model3.feature_importances_
    pred_test_y3 = model3.predict(test_x)
    print("The r2 score of AdaBoostRegression is :")
    print(sm.r2_score(test_y, pred_test_y3)) 
    
    
     #DecisionTreeRegressor                                 
    model2 = st.DecisionTreeRegressor(max_depth=4)
    #Train
    model2.fit(train_x,train_y)
    fi_dt = model2.feature_importances_
    #Predict
    pred_test_y2 = model2.predict(test_x)
    print("The r2 score of DecisionTreeRegression is :") 
    print(sm.r2_score(test_y, pred_test_y2)) 
    
     #RandomForestRegressor
    model = se.RandomForestRegressor(max_depth=10, n_estimators=1000, min_samples_split=2)
    model.fit(train_x, train_y)
    #feature importance
    fi_dy = model.feature_importances_
    #Test if the model is fitting well
    pred_test_y = model.predict(test_x)
    print("The r2 score of RandomForestRegresson is :")
    print(sm.r2_score(test_y, pred_test_y))
    
    
    