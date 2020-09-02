import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri
import statsmodels.formula.api as smf  # fitting regression models
import statsmodels.api as sm  # some regression model functions
import scipy.stats as sci  # some other useful math functions
import statsmodels.graphics.api as smg
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn import metrics
import numpy as np

df = pd.read_csv("BreastCancer.txt", sep=' ')

print(df.head())

df['Malignant'] = (df['Malignant']=="Yes")*1
# cols = df.columns.tolist()
# print(df.columns.tolist())
cols = ['Adhes', 'BNucl', 'Chrom', 'Epith', 'Mitos', 'NNucl', 'Thick', 'UShap', 'USize', 'Malignant']
df = df[cols]
# print(df.columns.tolist())

# print(df.head())

uneditedDF = df
# df = df[df.bmi != 0]
# df = df[df.diastolic != 0]
# df = df[df.triceps != 0]
# df = df[df.glucose != 0]
# df = df[df.age != 81]
minValuesObj = df.min()

print('minimum value in each column : ')
print(minValuesObj)

# sns.regplot(x='Adhes', y='Malignant', data=df, x_jitter=False, y_jitter=False, logistic=True)
# # plt.title('Age vs Diabetes')
#
# plt.show()
# sns.regplot(x='BNucl', y='Malignant', data=df, x_jitter=False, y_jitter=False, logistic=True)
# # plt.title('BMI vs Diabetes')
# plt.show()

pandas2ri.activate()
base_r = importr('base')
bestglm = importr('bestglm')
stats = importr('stats')

print(df.head())
# select = bestglm.bestglm(df, family=stats.binomial)
select = bestglm.bestglm(df, IC="AIC", method="exhaustive", family = stats.binomial)

print('CHECKKKKKKKKKKKKKK')
print(base_r.summary(select.rx2('BestModel')))

logit_reg = smf.glm(formula='Malignant~Adhes+BNucl+Chrom+Mitos+NNucl+Thick+UShap', data=df,
                    family=sm.families.Binomial()).fit()

print(logit_reg.summary())

fpr, sens, th = roc_curve(df['Malignant'], logit_reg.predict())

plt.plot(fpr, sens)  # plots the ROC curve
# plt.plot([0,1],[0,1],'k') #adds a 1-1 line for reference
plt.xlabel('False Positive Rate')  # adds an x label.
plt.ylabel('Sensitivity')  # adds an y label.
plt.title('ROC Curve')
plt.show()

print('AUC:', auc(fpr, sens))
print('Psuedo R^2: ', 1 - (logit_reg.deviance / logit_reg.null_deviance))

pred_probs = logit_reg.fittedvalues
thresh = np.linspace(0, 1, num=100)
misclass = np.repeat(np.NaN, repeats=thresh.size)

for i in range(thresh.size):
    my_classification = pred_probs > thresh[i]
    misclass[i] = np.mean(df['Malignant'] != my_classification)

threshold = thresh[np.argmin(misclass)]
# Find threshold which minimizes misclassification
print(threshold)

# print(type(logit_reg.predict()))
predictionArray = []
# logit_regMain = smf.glm(formula='diabetes~pregnant+glucose+bmi+pedigree', data=uneditedDF,
#                         family=sm.families.Binomial()).fit()
#
# fpr, sens, th = roc_curve(uneditedDF['diabetes'], logit_regMain.predict())
#
# print('AUC:', auc(fpr, sens))
# print('Psuedo R^2: ', 1 - (logit_regMain.deviance / logit_regMain.null_deviance))

for prediction in logit_reg.predict():
    if prediction > threshold:
        predictionArray.append(1)
    else:
        predictionArray.append(0)
cm = metrics.confusion_matrix(df['Malignant'], predictionArray)
print(cm)

tn, fp, fn, tp = cm.ravel()
print('Sensitivity: ', tp / (tp + fn))
print('Specitivity: ', tn / (tn + fp))
print('PPV: ', tp / (tp + fp))
print('NPV: ', tn / (tn + fn))


# Adhes= 3, BNucl= 1, Chrom= 5, Epith= 8, Mitos= 1, NNucl= 8, Thick= 3,UShap= 1
dframe = pd.DataFrame(dict(Adhes=[3], BNucl=[1], Chrom=[5], Epith=[8], Mitos=[1], NNucl=[8],
                           Thick=[3], UShap=[1]))
preds = logit_reg.get_prediction(dframe)


print('Prediction')
print(logit_reg.predict(dframe))

df = uneditedDF
n_cv = 500
numDataPoints = int(df.shape[0])
print(numDataPoints)
n_test = int(np.round(.1 * numDataPoints))
## Set my threshold for classifying
cutoff = threshold
## Initialize matrices to hold CV results
sens = np.repeat(np.NaN, n_cv)
spec = np.repeat(np.NaN, n_cv)
ppv = np.repeat(np.NaN, n_cv)
npv = np.repeat(np.NaN, n_cv)
auc = np.repeat(np.NaN, n_cv)

## Begin for loop
for cv in range(n_cv):
    ## Separate into test and training sets
    test_obs = np.random.choice(numDataPoints, n_test)
    # print(test_obs)

    # Split data into test and training sets
    test_set = df.iloc[test_obs, :]
    train_set = df.drop(test_obs, axis=0)

    ## Fit best model to training set
    train_model= smf.glm(formula='Malignant~Adhes+BNucl+Chrom+Mitos+NNucl+Thick+UShap', data=df,
                        family=sm.families.Binomial()).fit()

    ## Use fitted model to predict test set
    pred_probs = train_model.predict(test_set)

    ## Classify according to threshold
    test_class = pred_probs > cutoff

    ## Create a confusion matrix
    conf_mat = confusion_matrix(test_set['Malignant'], test_class)

    ## Pull of sensitivity, specificity, PPV and NPV
    ## using bracket notation
    tn, fp, fn, tp = conf_mat.ravel()
    sens[cv] = tp / (tp + fn)
    spec[cv] = tn / (tn + fp)
    ppv[cv] = tp / (tp + fp)
    npv[cv] = tn / (tn + fn)

    ## Calculate AUC
    fpr, tpr, th = roc_curve(test_set['Malignant'], pred_probs)
    # auc_value[cv] = auc(fpr, tpr)

print('SENSITIVITY: ', np.mean(sens))
print('SPECIFICITY: ', np.mean(spec))
print('PPV: ', np.mean(ppv))
print('NPV: ', np.mean(npv))