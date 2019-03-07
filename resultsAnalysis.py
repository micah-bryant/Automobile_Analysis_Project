import numpy as np
import matplotlib.pyplot as plt
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from dataOrg import scaler, invScale
from dataAnalysis import residuals_ind, residuals_full, calcPred, OLS_ind, OLS_full

def R2YCalc_ind(values,data,column,order):
    newData = scaler(data)
    residuals = residuals_ind(values,newData,column,order)
    for i in range(len(residuals)):
        residuals[i] = residuals[i]**2
    r2y = 1 - np.sum(residuals)/np.sum(newData[:,0]**2)
    return r2y

def R2YCalc_full(values,data):
    newData = scaler(data)
    residuals = residuals_full(values,newData)
    for i in range(len(residuals)):
        residuals[i] = residuals[i]**2
    r2y = 1 - np.sum(residuals)/np.sum(newData[:,0]**2)
    return r2y

def dataPlot_ind(values,data,column,order):
    if column == 1:
        label = "Cylinders"
    elif column == 2:
        label = "Displacement"
    elif column == 3:
        label = "Horsepower"
    elif column == 4:
        label = "Weight"
    elif column == 5:
        label = "Acceleration"
    newData = scaler(data)
    plt.figure()
    plt.scatter(data[:,column],data[:,0])
    plt.title('Measured Data')
    plt.ylabel("MPG")
    plt.xlabel(label)
    
    predVal = calcPred(values,newData,column,order)
    predVal = invScale(data,predVal)
    
    plt.figure()
    plt.scatter(data[:,column],predVal)
    plt.title('Predicted Data')
    plt.ylabel("MPG")
    plt.xlabel(label)
    
def OLS_LOOCV(data,column,order):
    R2Y = 0
    predVal = []
    for i in range(len(data[:,0])):
        train = np.zeros((len(data[:,0])-1,8))
        test = np.zeros((1,8))
        for j in range(len(data[:,0])):
            if j < i:
                train[j,:] = data[j,:]
            elif j > i:
                train[j-1,:] = data[j,:]
            else:
                test[0,:] = data[j,:]
        opt = OLS_ind(train,column,order)
        trainScale = StandardScaler()
        trainScale.fit(train)
        testScaled = trainScale.transform(test)
        pred = calcPred(opt,testScaled,column,order)
        predVal.append(np.squeeze(pred))
    scaledData = scaler(data)
    R2Y = 1-np.sum((predVal-scaledData[:,0])**2)/np.sum(scaledData[:,0]**2)
    return R2Y

def OLS_groupCV(data,column,order):
    R2Y = 0
    diff = []
    cities = [1,2,3]
    for group in cities:
        train = []
        test = []
        for i in range(len(data[:,0])):
            if data[i,7] == group:
                test.append(data[i,:])
            else:
                train.append(data[i,:])
        test = np.array(test)
        train = np.array(train)
        opt = OLS_ind(train,column,order)
        
        trainScale = StandardScaler()
        trainScale.fit(train)
        testScaled = trainScale.transform(test)
        pred = calcPred(opt,testScaled,column,order)
        
        totalScale = StandardScaler()
        totalScale.fit(data)
        scaledData = totalScale.transform(test)
        diff.append(np.sum((pred-scaledData[:,0])**2)/np.sum(scaledData[:,0]**2))
    return diff

def PLSR_LOOCV(data):
    R2Y = 0
    predVal = []
    for i in range(len(data[:,0])):
        train = np.zeros((len(data[:,0])-1,8))
        test = np.zeros((1,8))
        for j in range(len(data[:,0])):
            if j < i:
                train[j,:] = data[j,:]
            elif j > i:
                train[j-1,:] = data[j,:]
            else:
                test[0,:] = data[j,:]
        
        testScaled = np.zeros((1,8))
        trainScale = StandardScaler()
        trainScaled = trainScale.fit_transform(train)
        testScaled[0,:] = trainScale.transform(test)
        PLSR = PLSRegression(n_components = 2)
        PLSR.fit(trainScaled[:,2:6],trainScaled[:,0])
        pred = PLSR.predict(testScaled[:,2:6])
        predVal.append(np.squeeze(pred))
    scaledData = scaler(data)
    R2Y = 1-np.sum((predVal-scaledData[:,0])**2)/np.sum(scaledData[:,0]**2)
    return R2Y

def PLSR_groupCV(data):
    R2Y = 0
    diff = []
    cities = [1,2,3]
    for group in cities:
        train = []
        test = []
        for i in range(len(data[:,0])):
            if data[i,7] == group:
                test.append(data[i,:])
            else:
                train.append(data[i,:])
        test = np.array(test)
        train = np.array(train)
        
        trainScale = StandardScaler()
        trainScaled = trainScale.fit_transform(train)
        testScaled = trainScale.transform(test)
        PLSR = PLSRegression(n_components = 2)
        PLSR.fit(trainScaled[:,2:6],trainScaled[:,0])
        error = PLSR.score(testScaled[:,2:6],testScaled[:,0])
        diff.append(error)
    return diff