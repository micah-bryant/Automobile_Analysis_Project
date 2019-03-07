import numpy as np
import matplotlib.pyplot as plt
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from scipy.optimize import least_squares
from dataOrg import scaler

#PLSR
def func_PLSR(data): #needs the data to be scaled before hand (makes it easier to do cross validation)
    newData = scaler(data)
    PLSR = PLSRegression(n_components = 2)
    PLSR.fit(newData[:,1:6],newData[:,0])
    print('The R2Y value is', PLSR.score(newData[:,1:6],newData[:,0]))
    Xscores = PLSR.x_scores_
    Yscores = PLSR.y_scores_
    Xload = PLSR.x_loadings_
    Yload = PLSR.y_loadings_
    plt.figure()
    plt.scatter(Xscores[:,0],Xscores[:,1])
    plt.scatter(Yscores[:,0],Yscores[:,1])
    plt.title('Scores Plot')
    plt.figure()
    plt.scatter(Xload[0,0],Xload[0,1],label = 'Cylinders')
    plt.scatter(Xload[1,0],Xload[1,1],label = 'Displacement')
    plt.scatter(Xload[2,0],Xload[2,1],label = 'Horsepower')
    plt.scatter(Xload[3,0],Xload[3,1],label = 'Weight')
    plt.scatter(Xload[4,0],Xload[4,1],label = 'Acceleration')
    plt.scatter(Yload[:,0],Yload[:,1], label = 'MPG')
    plt.title('Loadings Plot')
    plt.legend(loc = 'best');

def cyl_PLSR(data):
    newData = scaler(data)
    PLSR = PLSRegression(n_components = 2)
    PLSR.fit(newData[:,2:6],newData[:,0])
    print('The R2Y value is', PLSR.score(newData[:,2:6],newData[:,0]))
    Xscores = PLSR.x_scores_
    Yscores = PLSR.y_scores_
    Xload = PLSR.x_loadings_
    Yload = PLSR.y_loadings_
    plt.figure()
    plt.scatter(Xscores[:,0],Xscores[:,1])
    plt.scatter(Yscores[:,0],Yscores[:,1])
    plt.title('Scores Plot')
    plt.figure()
    plt.scatter(Xload[0,0],Xload[0,1],label = 'Displacement')
    plt.scatter(Xload[1,0],Xload[1,1],label = 'Horsepower')
    plt.scatter(Xload[2,0],Xload[2,1],label = 'Weight')
    plt.scatter(Xload[3,0],Xload[3,1],label = 'Acceleration')
    plt.scatter(Yload[:,0],Yload[:,1], label = 'MPG')
    plt.title('Loadings Plot')
    plt.legend(loc = 'best');
    
#Least Squares
def calcPred(values,data,column,order):
    if order == 1:
        beta,offset = values
    elif order == 2:
        beta1,beta2,offset = values
    elif order == 3:
        beta1,beta2,beta3,offset = values
    elif order == 4:
        beta1,beta2,beta3,beta4,offset = values
    predVal = []
    for i in range(len(data[:,0])):
        if order == 1:
            prediction = (beta*data[i,column]) + offset
        elif order == 2:
            prediction = (beta1*(data[i,column]**2)) + (beta2*data[i,column]) + offset
        elif order == 3:
            prediction = (beta1*(data[i,column]**3)) + (beta2*(data[i,column]**2)) + (beta3*data[i,column]) + offset
        elif order == 4:
            prediction = (beta1*(data[i,column]**4)) + (beta2*(data[i,column]**3)) + (beta3*(data[i,column]**2)) + (beta4*data[i,column]) + offset
        predVal.append(prediction)
    predVal = np.array(predVal)
    return predVal

def residuals_ind(values,data,column, order):
    predVal = calcPred(values,data,column,order)
    residuals = predVal - data[:,0]
    return residuals

def residuals_full(values,data):
    beta1,beta2,beta3,beta4,beta5,offset = values
    predVal = []
    for i in range(len(x[:,0])):
        prediction = (beta1*data[i,1]) + (beta2*data[i,2]) + (beta3*data[i,3]) + (beta4*data[i,4]) + (beta5*data[i,5]) + offset
        predVal.append(prediction)
    predVal = np.array(predVal)
    residuals = predVal - data[:,0]
    return residuals

def OLS_ind(data,column,order):
    if order == 1:
        y0 = np.zeros(2)
    elif order == 2:
        y0 = np.zeros(3)
    elif order == 3:
        y0 = np.zeros(4)
    elif order == 4:
        y0 = np.zeros(5)
    newData = scaler(data)
    opt = least_squares(residuals_ind, y0, args = (newData,column,order))
    return opt.x

def OLS_full(data):
    y0 = np.zeros(6)
    data = scaler(data)
    opt = least_squares(residuals_full,y0,args = (data,))
    return opt.x