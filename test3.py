from pandas import read_csv
from pandas.tools.plotting import  scatter_matrix
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
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
#read data
filename = 'temp.csv'
dataset = read_csv(filename)

array = dataset.values
x = array[:136,0:1]
y = array[:,2]

prediction = np.array([2017])
lab_enc = preprocessing.LabelEncoder()
#
validation_size = 20
#
seed = 7
xtrain, xvalidation, ytrain, yvalidation = train_test_split(x, y, test_size=validation_size, random_state=seed)
  #
p = LogisticRegression()
p.fit(xtrain, ytrain)
# kfold = KFold(n_splits=10, random_state=seed)
# cv_result = cross_val_score( p,xtrain, ytrain, cv=kfold, scoring='accuracy')
#
#
# print(p.predict(prediction))