# -*- coding: utf-8 -*-
"""
Created on Thu Jun 28 10:35:30 2018

@author: Aastha
"""

''' step1:pick random k data from training set
    step2:build the decision tree associated to those k data points
    step3:choose the numbers of n transyou want to build and repeat step1 and step2
    step4:for a new data point make each one of your n-trees predict the value of y for the data point in the question and assign the new data point the average across all of the predicted y values'''
    
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    
    data=pd.read_csv('Position_Salaries.csv')
    x=data.iloc[:,1:2].values
    y=data.iloc[:,2].values
    
    from sklearn.ensemble import RandomForestRegressor
    regressor=RandomForestRegressor(n_estimators=10,random_state=0)
    regressor.fit(x,y)

    #predicting the value
    y_pred=regressor.predict(7)


    x_grid=np.arange(min(x),max(x),0.1)
    x_grid=x_grid.reshape((len(x_grid),1))
    plt.scatter(x,y,color='red')
    plt.plot(x_grid,regressor.predict(x_grid),color='blue')