from flask import Flask, render_template, request,redirect,url_for
import pickle
import numpy as np
from xlwt import Workbook
from  xlrd import open_workbook
import openpyxl
import os

#cleaning text
import re
import nltk
#nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
app = Flask(__name__)


@app.route('/welcome/<int:results>')
def welcome(results):
	if(results==1):
		return 'Thanks for your POSITIVE comment'
	else :
		return 'comment is NEGATIVE we would try to improve'

	
def modify(review1):
	f= open("comments","a+")
	f.write(review1)
	f.write("\n")
	f.close()
	
	

review="global"
review1=" "

@app.route('/input',methods=['POST','GET'])
def input():
	review=request.form['comment']
	review1=review
	model = pickle.load(open('model.pkl','rb'))
	cv=pickle.load(open('model1.pkl','rb'))
	from functions import clean
	review=clean(review)
	data=[review]
	vect = cv.transform(data).toarray()
	a=model.predict(vect)
	modify(review1)
	return redirect(url_for('welcome',results=a))  

    
       
if __name__ == '__main__':
	app.run(debug = True)
       


	

