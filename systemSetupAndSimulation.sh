#!/usr/bin/env python3

import keras
import pandas as pd
import numpy as np

import preprocessdata
import nnmodelsV4
import offline

from preprocessdata import Activities, IMUSensors
from nnmodelsV4 import NNModel, LSTMModelFactory, BaseCallbacksListFactory, loadNNModel
from offline import ActivityModule, Classifier, Sensor, ArgMinStrategy, SingleSensorSystem, PyPlotter

# reload modules (useful for development)
import importlib
importlib.reload(preprocessdata)
importlib.reload(nnmodelsV4)
importlib.reload(offline)

from preprocessdata import Activities, IMUSensors
from nnmodelsV4 import NNModel, LSTMModelFactory, BaseCallbacksListFactory, loadNNModel
from offline import ActivityModule, Classifier, Sensor, ArgMinStrategy, SingleSensorSystem, PyPlotter, Reasoner, MostFrequentStrartegy

baseDir = 'NNModels'   # NNModels base directory

#*****************************************************************************
#SYSTEM SPECIFICS
person = 'S1'
session ='ADL1'

activityCategory = 'mlBothArms' # 'locomotion' or 'mlBothArms'
sensorNames = ['backImu' , 'rlaImu', 'ruaImu', 'llaImu', 'luaImu']

lookback = 30
sensorChannels = 6

# Activities (get all activityNames)
activities = Activities()
activityNames = activities.getNamesWithCategory(activityCategory)

# Select subsamples of activities (only for 'mlBothArms' activity Category)
ind = np.array([0,2,4,5,6,7,8,9,14,15,16])
activityNames = np.array(activityNames)[ind]

print('\nSELECTED activity names:\n', activityNames)

classifyStrategy = ArgMinStrategy()
selectionStartegy = MostFrequentStrartegy()

#*****************************************************************************
# SYSTEM SETUP
identifier = {
               'activityCategory' : activityCategory,
               'sensor' : None,
               'activityName' : None,
            }

# setup of the sensorSystems and the corresponding Classifiers
sensorSystems = []
classifiers = []
for sensorName in sensorNames:
    identifier['sensor'] = sensorName
    activityModules = []
    for activityName in activityNames:
        identifier['activityName'] = activityName
        nnModel = loadNNModel(identifier, lookback = lookback, sensorChannels = sensorChannels, baseDir = baseDir)
        activityModules.append(ActivityModule(nnModel))
    sensorSystems.append(SingleSensorSystem(activityModules))
    classifiers.append(Classifier(classifyStrategy))

# setup of the Sensors
imuSensorsDataFrame = pd.read_csv('IMUSsensorsWithQuaternions.csv', header = [0,1], index_col = [0,1,2])
imuSensors = IMUSensors(imuSensorsDataFrame)
idx = pd.IndexSlice
identifier.pop('activityName')   # remove activityName from keys
sensors = []
for sensorName in sensorNames:
    identifier['sensor'] = sensorName
    sensorDf = imuSensors.singleSensorDf(sensorName).loc[idx[person, session], :]
    sensors.append(Sensor(sensorDf, identifier = identifier, sensorChannels = sensorChannels))

# setup aggregation Module
reasoner = Reasoner(selectionStartegy)

#*****************************************************************************
# SIMULATION
tiSim = 3000   #3000 times step (30 Hz -> 30 steps per second) sec = timsteps / freq
tfSim = 3150 #45000  times step (30 Hz -> 30 steps per second) sec = timsteps / freq

selectedActivityName = np.empty((tfSim-tiSim, len(sensors)), dtype=object)
resultantActivityName = np.empty(tfSim-tiSim, dtype=object)
sensorSystemsErrors = np.empty((len(sensors), tfSim-tiSim, len(activityNames)))
for t in range(tiSim, tfSim):
    for i in range(len(sensors)):
        sensorData = sensors[i].getDataWithTimeIndex(t)
        errorsAndIds = sensorSystems[i].getErrorsAndIds(sensorData) # errorAndIds is list of (float, dictionary)
        for j in range(len(activityNames)):
            sensorSystemsErrors[i,t - tiSim,j] =  errorsAndIds[j][0]   #  t timestep, i sensor system, , j activity module
        selectedActivityName[t - tiSim,i] = classifiers[i].classify(errorsAndIds)['activityName']
    resultantActivityName[t - tiSim] = reasoner.selectActivity(selectedActivityName[t - tiSim,:])

#*****************************************************************************
# PLOT
tiPlot = 3030 # times step (30 Hz -> 30 steps per second) sec = timsteps / freq
tfPlot = 3120 # times step (30 Hz -> 30 steps per second) sec = timsteps / freq

pyplotter = PyPlotter(tiSim, tfSim, activityCategory, imuSensorsDataFrame, person, session)
for i in range(len(sensorSystems)):
    pyplotter.plotSensorSystemErrors(sensorSystems[i], 
                                     sensorSystemsErrors[i,:,:], 
                                     selectedActivityName[:,i], 
                                     tiPlot, tfPlot, 
                                     figsize = (20,5), top = 2, 
                                     toFile = True)

pyplotter.plotSelectedVsTrue(resultantActivityName, 
                             tiPlot, tfPlot, 
                             figsize = (20,5), 
                             top = 2, 
                             toFile = True)








        