
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#cleaning text
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer


def clean(review) :
	review=re.sub('[^a-zA-Z]',' ',review)  
	review=review.lower()
	review=review.split()
	ps=PorterStemmer()  
	review=[ps.stem(word) for word in review if not word in set(stopwords.words('english'))] 
	review=' '.join(review)	
	return review



	