import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#import functions as f1

#cleaning text
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

dataset=pd.read_csv('Restaurant_Reviews.tsv',delimiter='\t',quoting=3)

corpus=[]
for i in range(0,1000):
	review=re.sub('[^a-zA-Z]',' ',dataset['Review'][i])  
	review=review.lower()
	review=review.split()
	ps=PorterStemmer()  
	review=[ps.stem(word) for word in review if not word in set(stopwords.words('english'))] 
	review=' '.join(review)
	corpus.append(review)
	#dataset['Review'][i]=f1.clean(dataset['Review'][i])
  


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

import pickle
pickle.dump(classifier, open('model.pkl','wb'))

pickle.dump(cv,open('model1.pkl','wb'))