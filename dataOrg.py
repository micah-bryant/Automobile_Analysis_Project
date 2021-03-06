import numpy as np
from sklearn.preprocessing import StandardScaler

#0 = MPG
#1 = Cyclinders
#2 = Displacement
#3 = Horsepower
#4 = Weight
#5 = Acceleration
#6 = Model Year (70-82)
#7 = City of Origin (1,2,3)

def textFileParse(filename):
    ''' Returns 2D matrix of data '''
    data = np.loadtxt(filename, usecols = (0,1,2,3,4,5,6,7))
    return data

def splitYear(data, year):
    ''' returns 2D matrix of the data that matches the year'''
    newData = []
    for i in range(len(data[:,0])):
        if data[i,6] == year:
            newData.append(data[i,:])
    newData = np.array(newData, dtype = 'float64')
    return newData

def splitCylinder(data):
    ''' Returns a list of 2D matrices organized by data corresponding to a # of cylinders '''
    newData = []
    cylinder = [4,6,8]
    for numCyl in cylinder:
        dataPlane = []
        for i in range(len(data[:,0])):
            if data[i,1] == numCyl:
                dataPlane.append(data[i,:])
        dataPlane = np.array(dataPlane)
        newData.append(dataPlane)
    return newData

def splitCat(data,column):
    ''' Returns a single column of data '''
    newData = data[:,column]
    return newData

def scaler(data):
    ''' Returns the Z-scored data '''
    newData = []
    for i in range(6):
        scaler = StandardScaler()
        values = np.zeros((len(data[:,i]),1))
        values[:,0] = data[:,i]
        values = scaler.fit_transform(values)
        newData.append(values[:,0])
    newData = np.array(newData)
    return np.transpose(newData)

def invScale(data,data_scaled):
    ''' Returns the actual predicted MPG when given Z-scored data '''
    scaler = StandardScaler()
    values = np.zeros((len(data[:,0]),1))
    values[:,0] = data[:,0]
    scaler.fit(values)
    scaledVal = np.zeros((len(data[:,0]),1))
    scaledVal[:,0] = data_scaled
    scaledVal = scaler.inverse_transform(scaledVal)
    return scaledVal

def valConv(values,data,column):
    ''' Returns the constants for the equation for actual data instead of Z-scored data '''
    C1,C2,C3 = values
    meanA = np.mean(data[:,column])
    meanB = np.mean(data[:,0])
    stdA  = np.std(data[:,column])
    stdB  = np.std(data[:,0])
    const1 = (C1*stdB)/(stdA**2)
    const2 = (C2/stdA-2*C1*meanA/stdA**2)*stdB
    const3 = stdB*(C1*meanA**2/stdA**2-C2*meanA/stdA+C3) + meanB
    return const1,const2,const3