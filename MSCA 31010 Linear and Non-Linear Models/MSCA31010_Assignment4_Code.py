# -*- coding: utf-8 -*-


import sys
import itertools
import matplotlib.pyplot as plt
import numpy
import pandas
import sklearn.neural_network as nn
import time
from scipy.stats import chi2
import Regression

numpy.set_printoptions(precision = 10, threshold = sys.maxsize)
numpy.set_printoptions(linewidth = numpy.inf)

pandas.set_option('display.max_columns', None)
pandas.set_option('display.expand_frame_repr', False)
pandas.set_option('max_colwidth', None)
pandas.set_option('precision', 10)

pandas.options.display.float_format = '{:,.7}'.format

#1a

Homeown = pandas.read_excel('Homeowner_Claim_History.xlsx')
Homeown.head()
catName = ['f_primary_age_tier','f_primary_gender', 'f_marital', 'f_residence_location','f_fire_alarm_type', 'f_mile_fire_station', 'f_aoi_tier']
#intName = ['exposure']

nPredictor = len(catName)

Homeown['Severity']=Homeown['amt_claims']/Homeown['num_claims']
trainData=Homeown.dropna().reset_index(drop = True)

data.boxplot(by="f_primary_age_tier", column="amt_claims", figsize=(10, 6), vert=False)
data.boxplot(by="f_primary_gender", column="amt_claims", figsize=(10, 6), vert=False)
data.boxplot(by="f_marital", column="amt_claims", figsize=(10, 6), vert=False)
data.boxplot(by="f_residence_location", column="amt_claims", figsize=(10, 6), vert=False)
data.boxplot(by="f_residence_location", column="amt_claims", figsize=(10, 6), vert=False)
data.boxplot(by="f_mile_fire_station", column="amt_claims", figsize=(10, 6), vert=False)

#1b

trainData = trainData[trainData['Severity']>0].reset_index(drop = True)
yName='Severity'
X0_train = trainData[[yName]].copy()
X0_train.insert(0, 'Intercept', 1.0)
X0_train.drop(columns = [yName], inplace = True)

y_train = trainData[yName].copy()

maxIter = 20
tolS = 1e-7
stepSummary = pandas.DataFrame()

resultList = Regression.GammaModel (X0_train, y_train, offset = None, maxIter = maxIter, tolSweep = tolS)

llk0 = resultList[3]
df0 = len(resultList[4])
stepSummary = stepSummary.append([['Intercept', ' ', df0, llk0, numpy.NaN, numpy.NaN, numpy.NaN]], ignore_index = True)
stepSummary.columns = ['Predictor', 'Type', 'ModelDF', 'ModelLLK', 'DevChiSq', 'DevDF', 'DevSig']

print('======= Step Detail =======')
print('Step = ', 0)
print('Step Statistics:')
print(stepSummary)

cName = catName.copy()
#iName = intName.copy()
entryThreshold = 0.05

for step in range(nPredictor):
    enterName = ''
    stepDetail = pandas.DataFrame()

    # Enter the next predictor
    for X_name in cName:
        X_train = pandas.get_dummies(trainData[[X_name]])
        X_train = X0_train.join(X_train)
        resultList = Regression.GammaModel (X_train, y_train, offset = None, maxIter = maxIter, tolSweep = tolS)
        llk1 = resultList[3]
        df1 = len(resultList[4])
        devChiSq = 2.0 * (llk1 - llk0)
        devDF = df1 - df0
        devSig = chi2.sf(devChiSq, devDF)
        stepDetail = stepDetail.append([[X_name, 'categorical', df1, llk1, devChiSq, devDF, devSig]], ignore_index = True)

    
    stepDetail.columns = ['Predictor', 'Type', 'ModelDF', 'ModelLLK', 'DevChiSq', 'DevDF', 'DevSig']

    # Find a predictor to enter, if any
    stepDetail.sort_values(by = ['DevSig', 'ModelLLK'], axis = 0, ascending = [True, False], inplace = True)
    enterRow = stepDetail.iloc[0].copy()
    minPValue = enterRow['DevSig']
    if (minPValue <= entryThreshold):
        stepSummary = stepSummary.append([enterRow], ignore_index = True)
        df0 = enterRow['ModelDF']
        llk0 = enterRow['ModelLLK']

        enterName = enterRow['Predictor']
        enterType = enterRow['Type']
        if (enterType == 'categorical'):
            X_train = pandas.get_dummies(trainData[[enterName]].astype('category'))
            X0_train = X0_train.join(X_train)
            cName.remove(enterName)
        elif (enterType == 'interval'):
            X_train = trainData[[enterName]]
            X0_train = X0_train.join(X_train)
            iName.remove(enterName)
    else:
        break

    # Print debugging output
    print('======= Step Detail =======')
    print('Step = ', step+1)
    print('Step Statistics:')
    print(stepDetail)
    print('Enter predictor = ', enterName)
    print('Minimum P-Value =', minPValue)
    print('\n')


    # Final model
resultList = Regression.GammaModel (X0_train, y_train, offset = None, maxIter = maxIter, tolSweep = tolS)

#outCoefficient = resultList[0]
alpha = resultList[7]
print("The estimate for the Shape parameter is", round(alpha, 7))



#1c

pandas.DataFrame(stepSummary)

#1d

# Final model
resultList = Regression.GammaModel (X0_train, y_train, offset = None, maxIter = maxIter, tolSweep = tolS)

outCoefficient = resultList[0]
alpha = resultList[7]
y_pred = resultList[6]

# Simple Residual
y_simple_residual = y_train - y_pred

# Mean Absolute Proportion Error
ape = numpy.abs(y_simple_residual) / y_train
mape = numpy.mean(ape)

# Root Mean Squared Error
mse = numpy.mean(numpy.power(y_simple_residual, 2))
rmse = numpy.sqrt(mse)

# Relative Error
relerr = mse / numpy.var(y_train, ddof = 0)

# Pearson correlation

corr_matrix = numpy.corrcoef(y_train, y_pred)
pearson_corr = corr_matrix[0,1]

# Pearson Residual
y_pearson_residual = y_simple_residual / numpy.sqrt(y_pred)
# Deviance Residual
r_vec = y_train / y_pred
di_2 = 2 * (r_vec - numpy.log(r_vec) - 1)
y_deviance_residual = numpy.where(y_simple_residual > 0, 1.0, -1.0) * numpy.sqrt(di_2)


print('Root Mean Squared Error =', round(rmse, 7))
print('Relative Error =', round(relerr, 7))
print('Mean Absolute Proportion Error =', round(mape, 7))
print('Pearson Correlation =', round(pearson_corr, 7))

#1e

fig, ax0 = plt.subplots(nrows = 1, ncols = 1, dpi = 200)
ax0.scatter(y_train, y_pred, c = 'royalblue', marker = 'o')
ax0.set_xlabel('')
ax0.set_ylabel('predicted Severity')
ax0.xaxis.grid(True)
ax0.yaxis.grid(True)
plt.show()

fig, ((ax0, ax1), (ax2, ax3)) = plt.subplots(nrows = 2, ncols = 2, dpi = 200, sharex = True,
                                             figsize = (24,12))
# Plot Absolute Proportion Errors versus observed Severity
ax0.scatter(y_train, ape, c = 'royalblue', marker = 'o')
ax0.set_xlabel('observed Severity')
ax0.set_ylabel('Absolute Proportion Errors')
ax0.xaxis.grid(True)
ax0.yaxis.grid(True)

# Plot simple residuals versus observed Severity
ax1.scatter(y_train, y_simple_residual, c = 'royalblue', marker = 'o')
ax1.set_xlabel('observed Severity')
ax1.set_ylabel('Simple Residual')
ax1.xaxis.grid(True)
ax1.yaxis.grid(True)

# Plot Pearson residuals versus observed Severity
ax2.scatter(y_train, y_pearson_residual, c = 'royalblue', marker = 'o')
ax2.set_xlabel('Observed Severity')
ax2.set_ylabel('Pearson Residual')
ax2.xaxis.grid(True)
ax2.yaxis.grid(True)

# Plot deviance residuals versus observed Severity
ax3.scatter(y_train, y_deviance_residual, c = 'royalblue', marker = 'o')
ax3.set_xlabel('Observed Severity')
ax3.set_ylabel('Deviance Residual')
ax3.xaxis.grid(True)
ax3.yaxis.grid(True)

plt.show()


#Q2

#a

trainData2 = Homeown[catName + [yName]].dropna().reset_index(drop = True)
n_sample = trainData2.shape[0]

X = pandas.get_dummies(trainData2[catName].astype('category'))
#X = X.join(trainData2[intName])
y = trainData2[yName]
#y_category = y.cat.categories
#n_category = len(y_category)
#y=y.astype(int)

# Grid Search for the best neural network architecture
result = pandas.DataFrame()
actFunc = ['relu','tanh']
nLayer = range(1,11,1)
nHiddenNeuron = range(1,6,1)

combList = itertools.product(actFunc, nLayer, nHiddenNeuron)

time_begin = time.time()
for comb in combList:
    actFunc = comb[0]
    nLayer = comb[1]
    nHiddenNeuron = comb[2]
    totalneuron=nHiddenNeuron*nLayer
    nnObj = nn.MLPRegressor(hidden_layer_sizes = totalneuron,
                            activation = actFunc, verbose = False,solver='lbfgs',
                            max_iter = 10000)
    thisFit = nnObj.fit(X, y)
   #y_predProb = pandas.DataFrame(nnObj.predict_proba(X), columns = y_category)
    y_pred = nnObj.predict(X)
    y_residual = y - y_pred
    ape = numpy.abs(y_residual) / y
    mape = numpy.mean(ape)

    result = result.append([[actFunc, nLayer, nHiddenNeuron,totalneuron, mape]], ignore_index = True)

result.columns = ['Activation Function', 'nLayer', 'nHiddenNeuron','totalNeuron','MAPE']
time_end = time.time()
elapsed_time = time_end - time_begin
elapsed_time

#b
# Locate the optimal architecture
optima_index = result['MAPE'].idxmin()
optima_row = result.iloc[optima_index]
actFunc = optima_row['Activation Function']
nLayer = optima_row['nLayer']
nHiddenNeuron = optima_row['nHiddenNeuron']
optima_row

#c
# Final model
bestmodel = nn.MLPRegressor(hidden_layer_sizes = (nHiddenNeuron,)*nLayer, activation = actFunc, verbose = False, solver = 'lbfgs', max_iter = 10000)
bestFit = bestmodel.fit(X, y)
y_pred = bestmodel.predict(X)

#outCoefficient = resultList[0]
#alpha = resultList[7]
#y_pred = resultList[6]

# Simple Residual
y_simple_residual = y - y_pred

# Mean Absolute Proportion Error
ape = numpy.abs(y_simple_residual) / y
mape = numpy.mean(ape)

# Root Mean Squared Error
mse = numpy.mean(numpy.power(y_simple_residual, 2))
rmse = numpy.sqrt(mse)

# Relative Error
relerr = mse / numpy.var(y, ddof = 0)

# Pearson Residual
y_pearson_residual = y_simple_residual / numpy.sqrt(y_pred)
# Pearson Correlation 
corr_matrix = numpy.corrcoef(y, y_pred)
pearson_corr = corr_matrix[0,1]

print('Root Mean Squared Error =', round(rmse, 7))
print('Relative Error =', round(relerr, 7))
print('Mean Absolute Proportion Error =', round(mape, 7))
print('Pearson Correlation =', round(pearson_corr, 7))

#d
# Plot predicted Severity versus observed Severity
fig, ax0 = plt.subplots(nrows = 1, ncols = 1, dpi = 200)
ax0.scatter(y, y_pred, c = 'royalblue', marker = 'o')
ax0.set_xlabel('')
ax0.set_ylabel('predicted Severity')
ax0.xaxis.grid(True)
ax0.yaxis.grid(True)
plt.show()

fig, (ax0,ax1, ax2) = plt.subplots(3, dpi = 200, sharex = True,
                                             figsize = (12,12))
# Plot Absolute Proportion Errors versus observed Severity
ax0.scatter(y, ape, c = 'royalblue', marker = 'o')
ax0.set_xlabel('observed Severity')
ax0.set_ylabel('Absolute Proportion Errors')
ax0.xaxis.grid(True)
ax0.yaxis.grid(True)

# Plot simple residuals versus observed Severity
ax1.scatter(y, y_simple_residual, c = 'royalblue', marker = 'o')
ax1.set_xlabel('observed Severity')
ax1.set_ylabel('Simple Residual')
ax1.xaxis.grid(True)
ax1.yaxis.grid(True)

# Plot Pearson residuals versus observed Severity
ax2.scatter(y, y_pearson_residual, c = 'royalblue', marker = 'o')
ax2.set_xlabel('Observed Severity')
ax2.set_ylabel('Pearson Residual')
ax2.xaxis.grid(True)
ax2.yaxis.grid(True)



plt.show()
