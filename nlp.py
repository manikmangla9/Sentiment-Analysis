import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import functions as f1

dataset=pd.read_csv('Restaurant_Reviews.tsv',delimiter='\t',quoting=3)


for i in range(0,1000):
	dataset['Review'][i]=f1.clean(dataset['Review'][i])
  


#creating bag of words
 import sklearn
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=1500)
X=cv.fit_transform(corpus).toarray()
Y=dataset.iloc[:,1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, Y_train)

Y_pred=classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(Y_test,Y_pred)