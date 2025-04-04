import matplotlib.pyplot as plt
import numpy
import pandas
import sys
from scipy.stats import chi2
import Regression
# Set some options for printing all the columns
numpy.set_printoptions(precision = 10, threshold = sys.maxsize)
numpy.set_printoptions(linewidth = numpy.inf)

pandas.set_option('display.max_columns', None)
pandas.set_option('display.expand_frame_repr', False)
pandas.set_option('max_colwidth', None)
pandas.set_option('precision', 10)

pandas.options.display.float_format = '{:,.7}'.format



data = pandas.read_csv("Telco-Customer-Churn.csv")
# data.reset_index(inplace = True)
def parse(x):
    try:
        x = float(x)
    except:
        x = numpy.nan
    return x
data["TotalCharges"] = data["TotalCharges"].apply(parse)
data = data.dropna()

data.head()

#Question 1

#a

xtab = pandas.crosstab(index = data['gender'], columns = data['Churn'])
xtab.reset_index(inplace = True)
xtab['N'] = xtab["Yes"] + xtab["No"]
xtab['Odds'] = xtab["Yes"] / xtab["No"]
xtab.sort_values(by = 'Odds', inplace = True)

plt.bar(xtab['gender'], xtab['Odds'], color = 'firebrick')
plt.xlabel('Gender')
plt.ylabel('Odds of Churn=Yes vs Churn=No')
plt.xticks(xtab['gender'])
plt.grid(axis ='y')
plt.show()

xtab = pandas.crosstab(index = data['SeniorCitizen'], columns = data['Churn'])
xtab.reset_index(inplace = True)
xtab['N'] = xtab["Yes"] + xtab["No"]
xtab['Odds'] = xtab["Yes"] / xtab["No"]
xtab.sort_values(by = 'Odds', inplace = True)

plt.bar(xtab['SeniorCitizen'], xtab['Odds'], color = 'firebrick')
plt.xlabel('SeniorCitizen')
plt.ylabel('Odds of Churn=Yes vs Churn=No')
plt.xticks(xtab['SeniorCitizen'])
plt.grid(axis ='y')
plt.show()

xtab = pandas.crosstab(index = data['Partner'], columns = data['Churn'])
xtab.reset_index(inplace = True)
xtab['N'] = xtab["Yes"] + xtab["No"]
xtab['Odds'] = xtab["Yes"] / xtab["No"]
xtab.sort_values(by = 'Odds', inplace = True)

plt.bar(xtab['Partner'], xtab['Odds'], color = 'firebrick')
plt.xlabel('Partner')
plt.ylabel('Odds of Churn=Yes vs Churn=No')
plt.xticks(xtab['Partner'])
plt.grid(axis ='y')
plt.show()

xtab = pandas.crosstab(index = data['Dependents'], columns = data['Churn'])
xtab.reset_index(inplace = True)
xtab['N'] = xtab["Yes"] + xtab["No"]
xtab['Odds'] = xtab["Yes"] / xtab["No"]
xtab.sort_values(by = 'Odds', inplace = True)

plt.bar(xtab['Dependents'], xtab['Odds'], color = 'firebrick')
plt.xlabel('Dependents')
plt.ylabel('Odds of Churn=Yes vs Churn=No')
plt.xticks(xtab['Dependents'])
plt.grid(axis ='y')
plt.show()

xtab = pandas.crosstab(index = data['PhoneService'], columns = data['Churn'])
xtab.reset_index(inplace = True)
xtab['N'] = xtab["Yes"] + xtab["No"]
xtab['Odds'] = xtab["Yes"] / xtab["No"]
xtab.sort_values(by = 'Odds', inplace = True)

plt.bar(xtab['PhoneService'], xtab['Odds'], color = 'firebrick')
plt.xlabel('PhoneService')
plt.ylabel('Odds of Churn=Yes vs Churn=No')
plt.xticks(xtab['PhoneService'])
plt.grid(axis ='y')
plt.show()

xtab = pandas.crosstab(index = data['MultipleLines'], columns = data['Churn'])
xtab.reset_index(inplace = True)
xtab['N'] = xtab["Yes"] + xtab["No"]
xtab['Odds'] = xtab["Yes"] / xtab["No"]
xtab.sort_values(by = 'Odds', inplace = True)

plt.bar(xtab['MultipleLines'], xtab['Odds'], color = 'firebrick')
plt.xlabel('MultipleLines')
plt.ylabel('Odds of Churn=Yes vs Churn=No')
plt.xticks(xtab['MultipleLines'])
plt.grid(axis ='y')
plt.show()

xtab = pandas.crosstab(index = data['Contract'], columns = data['Churn'])
xtab.reset_index(inplace = True)
xtab['N'] = xtab["Yes"] + xtab["No"]
xtab['Odds'] = xtab["Yes"] / xtab["No"]
xtab.sort_values(by = 'Odds', inplace = True)

plt.bar(xtab['Contract'], xtab['Odds'], color = 'firebrick')
plt.xlabel('Contract')
plt.ylabel('Odds of Churn=Yes vs Churn=No')
plt.xticks(xtab['Contract'])
plt.grid(axis ='y')
plt.show()

xtab = pandas.crosstab(index = data['PaperlessBilling'], columns = data['Churn'])
xtab.reset_index(inplace = True)
xtab['N'] = xtab["Yes"] + xtab["No"]
xtab['Odds'] = xtab["Yes"] / xtab["No"]
xtab.sort_values(by = 'Odds', inplace = True)

plt.bar(xtab['PaperlessBilling'], xtab['Odds'], color = 'firebrick')
plt.xlabel('PaperlessBilling')
plt.ylabel('Odds of Churn=Yes vs Churn=No')
plt.xticks(xtab['PaperlessBilling'])
plt.grid(axis ='y')
plt.show()

#b

data.boxplot(column = 'TotalCharges', by = 'Churn', vert = False, figsize = (8,6))

data.boxplot(column = 'MonthlyCharges', by = 'Churn', vert = False, figsize = (8,6))

data.boxplot(column = 'tenure', by = 'Churn', vert = False, figsize = (8,6))

#Question2
#a
catName = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'Contract', 'PaperlessBilling']
intName = ['MonthlyCharges', 'TotalCharges', 'tenure']
yName = 'Churn'
nPredictor = len(catName) + len(intName)
trainData = data[[yName] + catName + intName].dropna()
trainData.head()

# Reorder the categories of the categorical variables in ascending frequency
for cat in catName:
    u = data[cat].astype('category')
    u_freq = u.value_counts(ascending = True)
    trainData[cat] = u.cat.reorder_categories(list(u_freq.index))


n_sample = trainData.shape[0]
X0_train = trainData[[yName]].copy()
X0_train.insert(0, 'Intercept', 1.0)
X0_train.drop(columns = [yName], inplace = True)
y_train = trainData[yName].apply(get_value)
def get_value(x):
    if x == "Yes":
        return 1
    else:
        return 0

maxIter = 20
tolS = 1e-7
stepSummary = pandas.DataFrame()
# Intercept only model
resultList = Regression.BLogisticModel (X0_train, y_train, offset = None, maxIter = maxIter, tolSweep = tolS)


llk0 = resultList[3]
df0 = len(resultList[4])
model_llk0 = resultList[3]
model_df0 = len(resultList[4])

stepSummary = stepSummary.append([['Intercept', ' ', df0, llk0, numpy.NaN, numpy.NaN, numpy.NaN]], ignore_index = True)
stepSummary.columns = ['Predictor', 'Type', 'ModelDF', 'ModelLLK', 'DevChiSq', 'DevDF', 'DevSig']

print('======= Step Detail =======')
print('Step = ', 0)
print('Step Statistics:')
print(stepSummary)


cName = catName.copy()
iName = intName.copy()
entryThreshold = 0.001

for step in range(nPredictor):
    enterName = ''
    stepDetail = pandas.DataFrame()

    # Enter the next predictor
    for X_name in cName:
        X_train = pandas.get_dummies(trainData[[X_name]])
        X_train = X0_train.join(X_train)
        resultList = Regression.BLogisticModel (X_train, y_train, offset = None, maxIter = maxIter, tolSweep = tolS)
        llk1 = resultList[3]
        df1 = len(resultList[4])
        devChiSq = 2.0 * (llk1 - llk0)
        devDF = df1 - df0
        devSig = chi2.sf(devChiSq, devDF)
        stepDetail = stepDetail.append([[X_name, 'categorical', df1, llk1, devChiSq, devDF, devSig]], ignore_index = True)

    for X_name in iName:
        X_train = trainData[[X_name]]
        X_train = X0_train.join(X_train)
        resultList = Regression.BLogisticModel (X_train, y_train, offset = None, maxIter = maxIter, tolSweep = tolS)
        llk1 = resultList[3]
        df1 = len(resultList[4])
        devChiSq = 2.0 * (llk1 - llk0)
        devDF = df1 - df0
        devSig = chi2.sf(devChiSq, devDF)
        stepDetail = stepDetail.append([[X_name, 'interval', df1, llk1, devChiSq, devDF, devSig]], ignore_index = True)

    stepDetail.columns = ['Predictor', 'Type', 'ModelDF', 'ModelLLK', 'DevChiSq', 'DevDF', 'DevSig']

    # Find a predictor to enter, if any
    stepDetail.sort_values(by = 'DevSig', axis = 0, ascending = True, inplace = True)
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

# End of forward selection
print('======= Step Summary =======')
print(stepSummary)

# Final model
resultList = Regression.BLogisticModel (X0_train, y_train, offset = None, maxIter = maxIter, tolSweep = tolS)

# Add interactions (to be completed by students)

#b
resultList[0]

#Question3

#a
model_llk1 = resultList[3]
model_df1 = len(resultList[4])

R_MF = 1.0 - (model_llk1 / model_llk0)

R_CS = numpy.exp(model_llk0 - model_llk1)
R_CS = 1.0 - numpy.power(R_CS, (2.0 / n_sample))

upbound = 1.0 - numpy.power(numpy.exp(model_llk0), (2.0 / n_sample))
R_N = R_CS / upbound


print('McFadden R2 = ', R_MF)
print('Cox-Snell R2 = ', R_CS)
print('Nagelkerke R2 = ', R_N)

predprob_event = resultList[6][1]
y = y_train.reset_index()["Churn"]
S1 = numpy.mean(predprob_event[y == 1])
S0 = numpy.mean(predprob_event[y == 0])

R_TJ = S1 - S0

print('Tjur R2 = ', R_TJ)


#b

def binary_model_metric (target, valueEvent, valueNonEvent, predProbEvent, eventProbThreshold = 0.5):
   '''Calculate metrics for a binary classification model

   Parameter
   ---------
   target: Panda Series that contains values of target variable
   valueEvent: Formatted value of target variable that indicates an event
   valueNonEvent: Formatted value of target variable that indicates a non-event
   predProbEvent: Panda Series that contains predicted probability that the event will occur
   eventProbThreshold: Threshold for event probability to indicate a success

   Return
   ------
   outSeries: Pandas Series that contain the following statistics
              ASE: Average Squared Error
              RASE: Root Average Squared Error
              MCE: Misclassification Rate
              AUC: Area Under Curve
   '''

   # Number of observations
   nObs = len(target)

   # Aggregate observations by the target values and the predicted probabilities
   aggrProb = pandas.crosstab(predProbEvent, target, dropna = True)

   # Calculate the root average square error
   ase = (numpy.sum(aggrProb[valueEvent] * (1.0 - aggrProb.index)**2) +
          numpy.sum(aggrProb[valueNonEvent] * (0.0 - aggrProb.index)**2)) / nObs
   if (ase > 0.0):
      rase = numpy.sqrt(ase)
   else:
      rase = 0.0

   # Calculate the misclassification error rate
   nFP = numpy.sum(aggrProb[valueEvent].iloc[aggrProb.index < eventProbThreshold])
   nFN = numpy.sum(aggrProb[valueNonEvent].iloc[aggrProb.index >= eventProbThreshold])
   mce = (nFP + nFN) / nObs

   # Calculate the number of concordant, discordant, and tied pairs
   nConcordant = 0.0
   nDiscordant = 0.0
   nTied = 0.0

   # Loop over the predicted event probabilities from the Event column
   predEP = aggrProb.index
   eventFreq = aggrProb[valueEvent]

   for i in range(len(predEP)):
      eProb = predEP[i]
      eFreq = eventFreq.loc[eProb]
      if (eFreq > 0.0):
         nConcordant = nConcordant + numpy.sum(eFreq * aggrProb[valueNonEvent].iloc[eProb > aggrProb.index])
         nDiscordant = nDiscordant + numpy.sum(eFreq * aggrProb[valueNonEvent].iloc[eProb < aggrProb.index])
         nTied = nTied + numpy.sum(eFreq * aggrProb[valueNonEvent].iloc[eProb == aggrProb.index])

   auc = 0.5 + 0.5 * (nConcordant - nDiscordant) / (nConcordant + nDiscordant + nTied)

   outSeries = pandas.Series({'ASE': ase, 'RASE': rase, 'MCE': mce, 'AUC': auc})
   return(outSeries)

outSeries = binary_model_metric (y, 1, 0, predprob_event, eventProbThreshold = 0.5)
print(outSeries)

#c
def f1_score(actual, predicted):
    tp = numpy.sum((actual==1) & (predicted==1))
    fp = numpy.sum((actual!=1) & (predicted==1))
    fn = numpy.sum((predicted!=1) & (actual==1))
    
    precision = tp/(tp+fp)
    recall = tp/(tp+fn)
    f1 = 2 * (precision * recall) / (precision + recall)
    return f1

thresholds=[0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.8, 0.9, 1.0]
scores = []
for threshold in thresholds:
    probs =predprob_event.copy()
    preds = probs.apply(lambda x: 1 if x >=threshold else 0)
    f1 = f1_score(y, preds)
    scores.append(f1)
plt.plot(thresholds, scores)

max(scores)

thresholds[scores.index(max(scores))]

probs =predprob_event.copy()
preds = probs.apply(lambda x: 1 if x >=0.3 else 0)
fp=preds[(preds==1)&(y==0)]
fn=preds[(preds==0)&(y==1)]
(len(fp)+len(fn))/len(preds)
