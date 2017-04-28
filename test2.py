from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from pandas import read_csv
from pandas.tools.plotting import  scatter_matrix
from matplotlib import pyplot
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from sklearn.metrics import  confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn import utils
from sklearn import preprocessing
import numpy as np

filename = 'iris.data.csv'
# names = ['Year', 'MonthOfYear',	'DayOfMonth',	'DayOfYear',	'AirTemperature(C)',	'AirTemperatureHygroClip(C)']

dataset = read_csv(filename)
# #visual
# scatter_matrix(dataset)
# pyplot.show()

#validation test

array = dataset.values

x = array[:,0:4]
y = array[:,4]


prediction = np.array([1, .3, .5, .09])
lab_enc = preprocessing.LabelEncoder()
# encoded = lab_enc.fit_transform(y)
validation_size = 20
seed = 7

xtrain, xvalidation, ytrain, yvalidation = train_test_split(x, y, test_size=validation_size, random_state=seed)

p = LogisticRegression()
p.fit(xtrain, ytrain)
kfold = KFold(n_splits=10, random_state=seed)
cv_result = cross_val_score( p,xtrain, ytrain, cv=kfold, scoring='accuracy')

# print( cv_result.mean(),cv_result.std() )

print(p.predict(prediction))





